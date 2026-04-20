"""
stt/constrained_decoder.py
============================
Constrained decoding for Whisper-based ASR.

Mathematical Formulation
========================

Standard Whisper beam search finds:

    w* = argmax_{w} sum_{t} log P_ASR(w_t | w_{<t}, X)

where X is the mel-spectrogram input.

─────────────────────────────────────────────────────────────────────────────
Logit Biasing
─────────────────────────────────────────────────────────────────────────────

We modify the raw decoder logits z_t (before softmax) at each step t:

    z'_t(v) = z_t(v) + alpha * LM_bias(v, context)

where:
    v           = vocabulary token index
    alpha       = bias strength hyperparameter  (default 2.0)
    LM_bias(v)  = log P_LM(word(v) | context)  if word(v) ∈ domain vocab
                = 0                             otherwise

This is equivalent to interpolating log-probabilities:

    log P_combined(v | X, h) ∝ log P_ASR(v | X, h) + alpha * log P_LM(v | h)

which in probability space gives a product-of-experts:

    P_combined ∝ P_ASR^1  *  P_LM^alpha

─────────────────────────────────────────────────────────────────────────────
Beam Search Modification
─────────────────────────────────────────────────────────────────────────────

Standard beam search maintains B hypotheses h_1, ..., h_B scored by:

    score(h) = sum_{t} log P_ASR(w_t | w_{<t}, X)

Constrained beam search replaces the acoustic score with the biased score:

    score_constrained(h) =
        sum_{t} [log P_ASR(w_t | w_{<t}, X)
                 + alpha * LM_bias(w_t, context_t)
                 - beta  * coverage_penalty(h)]

where:
    coverage_penalty(h) = sum over domain words not yet produced in h
                          (penalises hypotheses that skip required vocab)

    beta                = coverage penalty weight  (default 0.1)

─────────────────────────────────────────────────────────────────────────────
Constrained Candidate Forcing (optional hard constraint)
─────────────────────────────────────────────────────────────────────────────

For vocabulary that MUST appear, we implement prefix-tree forcing:
if a domain keyword k matches the beginning of the current hypothesis,
we force the beam to complete the keyword before considering other tokens.

    P_forced(v | h) = P_ASR(v | X, h)  if v completes keyword k
                    = 0                  otherwise

─────────────────────────────────────────────────────────────────────────────

class ConstrainedDecoder:
    - __init__(lm, alpha, beta, beam_size, hard_vocab)
    - build_token_bias_map(tokenizer, vocab)
    - bias_logits(logits, context_tokens, step)
    - beam_search(model, mel, language)
    - generate(model, mel, language, ...)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .ngram_lm import NgramLM

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Beam hypothesis dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeamHypothesis:
    """
    A single hypothesis in the beam.

    Fields
    ------
    token_ids       : decoded token IDs so far
    words           : decoded word strings (space-split from tokens)
    asr_score       : cumulative  sum log P_ASR
    lm_score        : cumulative  sum alpha * log P_LM
    combined_score  : asr_score + lm_score - beta * coverage_penalty
    domain_hits     : set of domain words already produced
    """
    token_ids:      List[int]    = field(default_factory=list)
    words:          List[str]    = field(default_factory=list)
    asr_score:      float        = 0.0
    lm_score:       float        = 0.0
    combined_score: float        = 0.0
    domain_hits:    Set[str]     = field(default_factory=set)
    is_finished:    bool         = False

    def score_per_token(self) -> float:
        n = max(len(self.token_ids), 1)
        return self.combined_score / n

    def clone(self) -> "BeamHypothesis":
        return BeamHypothesis(
            token_ids      = self.token_ids.copy(),
            words          = self.words.copy(),
            asr_score      = self.asr_score,
            lm_score       = self.lm_score,
            combined_score = self.combined_score,
            domain_hits    = self.domain_hits.copy(),
            is_finished    = self.is_finished,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedDecoder
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedDecoder:
    """
    Beam-search decoder for Whisper with N-gram language model logit biasing.

    Mathematical summary
    --------------------
    Modified logit at step t for token v:

        z'_t(v) = z_t(v) + alpha * LM_bias(v, context)

    Combined beam score:

        S(h) = sum_t [log P_ASR + alpha * log P_LM] - beta * coverage_penalty

    Parameters
    ----------
    lm             : trained NgramLM instance
    alpha          : LM logit bias strength  (default 2.0)
    beta           : coverage penalty weight (default 0.1)
    beam_size      : number of beams         (default 5)
    hard_vocab     : set of words that MUST be boosted regardless of LM score
    max_new_tokens : maximum generation length
    temperature    : sampling temperature (0 = greedy beam)
    min_lm_logprob : floor for LM log-prob (prevents -inf from collapsing beam)
    """

    def __init__(
        self,
        lm:              NgramLM,
        alpha:           float = 2.0,
        beta:            float = 0.1,
        beam_size:       int   = 5,
        hard_vocab:      Optional[Set[str]] = None,
        max_new_tokens:  int   = 448,
        temperature:     float = 0.0,
        min_lm_logprob:  float = -10.0,
    ):
        self.lm              = lm
        self.alpha           = alpha
        self.beta            = beta
        self.beam_size       = beam_size
        self.hard_vocab      = hard_vocab or lm.word_set
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.min_lm_logprob  = min_lm_logprob

        # Token-level bias map: token_id → bias_value (built lazily)
        # Maps Whisper tokenizer ID → LM log P(word)
        self._token_bias_map: Optional[Dict[int, float]] = None
        self._tokenizer       = None

        log.info(
            "ConstrainedDecoder | alpha=%.2f | beta=%.2f | beam=%d | "
            "vocab=%d words",
            alpha, beta, beam_size, len(self.hard_vocab),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Token bias map
    # ─────────────────────────────────────────────────────────────────────

    def build_token_bias_map(self, tokenizer) -> Dict[int, float]:
        """
        Pre-compute a bias value for every token in Whisper's vocabulary.

        For each token v:

            bias(v) = alpha * max(log P_LM(word(v)), min_lm_logprob)
                      if word(v) ∈ domain vocab
                    = 0   otherwise

        This map is applied as an additive offset to logits at every step,
        giving O(1) per-step overhead after the one-time build cost.

        Returns
        -------
        Dict[token_id → float]  — only non-zero entries are stored
        """
        self._tokenizer = tokenizer
        bias_map: Dict[int, float] = {}

        vocab = tokenizer.encoding._mergeable_ranks   # tiktoken vocab
        for token_bytes, token_id in vocab.items():
            try:
                word = token_bytes.decode("utf-8").strip().lower()
            except UnicodeDecodeError:
                continue

            if not word or word.startswith("<"):
                continue

            # Check if this token or its first word matches domain vocab
            first_word = word.split()[0]
            if first_word in self.hard_vocab or word in self.hard_vocab:
                lp = self.lm.log_prob(first_word)
                lp = max(lp, self.min_lm_logprob)
                bias = self.alpha * lp
                if bias != 0.0:
                    bias_map[token_id] = bias

        self._token_bias_map = bias_map
        log.info(
            "Token bias map built: %d tokens biased out of %d vocab",
            len(bias_map), len(vocab),
        )
        return bias_map

    # ─────────────────────────────────────────────────────────────────────
    # Logit biasing
    # ─────────────────────────────────────────────────────────────────────

    def bias_logits(
        self,
        logits:          torch.Tensor,
        context_tokens:  List[int],
        step:            int,
    ) -> torch.Tensor:
        """
        Apply LM bias to raw decoder logits.

        Mathematical operation:
            z'_t(v) = z_t(v) + bias(v)

        where bias(v) = alpha * log P_LM(word(v) | context)

        Parameters
        ----------
        logits         : (vocab_size,)  raw logits from Whisper decoder
        context_tokens : list of decoded token IDs so far
        step           : decoding step index (for logging)

        Returns
        -------
        z_prime : (vocab_size,)  biased logits
        """
        if self._token_bias_map is None:
            return logits   # bias map not built yet

        if not self._token_bias_map:
            return logits

        # Build sparse bias tensor
        z_prime = logits.clone()

        # Compute context words for dynamic LM scoring
        context_words = self._token_ids_to_words(context_tokens)

        for token_id, static_bias in self._token_bias_map.items():
            if token_id >= z_prime.shape[-1]:
                continue

            # Static bias (precomputed unigram)
            dynamic_bias = static_bias

            # Dynamic context-sensitive LM scoring (trigram override)
            if context_words and self.lm._nltk_model is not None:
                try:
                    word = self._tokenizer.decode([token_id]).strip().lower()
                    if word and word in self.hard_vocab:
                        ctx_lp = self.lm.log_prob(word, context_words[-2:])
                        ctx_lp = max(ctx_lp, self.min_lm_logprob)
                        dynamic_bias = self.alpha * ctx_lp
                except Exception:
                    pass   # keep static bias

            z_prime[token_id] += dynamic_bias

        return z_prime

    # ─────────────────────────────────────────────────────────────────────
    # Coverage penalty
    # ─────────────────────────────────────────────────────────────────────

    def _coverage_penalty(
        self,
        hypothesis:    BeamHypothesis,
        required_words: Set[str],
    ) -> float:
        """
        Coverage penalty encourages the beam to cover required domain words.

            penalty(h) = beta * |required_words - domain_hits(h)|
                                / |required_words|

        A fully-covering hypothesis has penalty = 0.
        A hypothesis that has produced none of the required words has
        penalty = beta.

        Parameters
        ----------
        hypothesis     : current BeamHypothesis
        required_words : domain words expected in this audio segment

        Returns
        -------
        float  penalty value (added negatively to the combined score)
        """
        if not required_words:
            return 0.0
        missing = required_words - hypothesis.domain_hits
        return self.beta * len(missing) / max(len(required_words), 1)

    # ─────────────────────────────────────────────────────────────────────
    # Beam search
    # ─────────────────────────────────────────────────────────────────────

    def beam_search(
        self,
        model,
        mel:             torch.Tensor,
        language:        str = "hi",
        required_words:  Optional[Set[str]] = None,
        device:          Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Constrained beam search over Whisper's decoder.

        At each step t:
        ─────────────────────────────────────────────────────────────────
        1. Forward pass: get logits z_t from Whisper decoder
        2. Apply LM bias:
               z'_t = z_t + alpha * LM_bias  (additive log-space mixing)
        3. Expand each beam hypothesis with top-beam_size tokens:
               candidates = top-k(softmax(z'_t))
        4. Score each candidate:
               S_new = S_old + log P_ASR(v) + alpha*log P_LM(v) - beta*coverage
        5. Retain top beam_size hypotheses by S_new
        ─────────────────────────────────────────────────────────────────

        Parameters
        ----------
        model          : Whisper model
        mel            : log-mel spectrogram  (n_mels, T)
        language       : "hi" or "en"
        required_words : domain words to encourage (for coverage penalty)
        device         : torch device string

        Returns
        -------
        (best_text, avg_log_prob)
        """
        try:
            import whisper
            from whisper.tokenizer import get_tokenizer
        except ImportError:
            raise ImportError("Run: pip install openai-whisper")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = get_tokenizer(multilingual=True, language=language, task="transcribe")

        # Build bias map if not yet done
        if self._token_bias_map is None:
            self.build_token_bias_map(tokenizer)

        # Special token IDs
        sot_id       = tokenizer.sot
        eot_id       = tokenizer.eot
        lang_id      = tokenizer.to_language_token(language)
        transcribe_id= tokenizer.transcribe
        no_ts_id     = tokenizer.no_timestamps

        # Initial prompt tokens
        init_tokens = [sot_id, lang_id, transcribe_id, no_ts_id]

        # ── FIX: resolve the hard context-window ceiling ──────────────────
        # Whisper's positional embedding table has exactly n_text_ctx rows
        # (448 for all published models).  token_tensor fed to the decoder
        # must NEVER exceed this length — otherwise the embedding lookup at
        # position n_text_ctx crashes with a tensor size mismatch.
        #
        # We subtract len(init_tokens) because those prompt tokens already
        # occupy the first slots of the context window.
        n_text_ctx = getattr(model.dims, "n_text_ctx", 448)   # 448 for all Whisper sizes
        effective_max = n_text_ctx - len(init_tokens)          # = 444 for standard models
        max_gen_steps = min(self.max_new_tokens, effective_max)
        # ─────────────────────────────────────────────────────────────────

        # Encode audio once
        with torch.no_grad():
            audio_features = model.encoder(mel.unsqueeze(0).to(device))

        # Initialise single beam
        beams: List[BeamHypothesis] = [
            BeamHypothesis(token_ids=init_tokens.copy())
        ]
        finished: List[BeamHypothesis] = []

        required_words = required_words or set()

        for step in range(max_gen_steps):           # was: self.max_new_tokens
            if not beams:
                break

            all_candidates: List[BeamHypothesis] = []

            for hyp in beams:
                if hyp.is_finished:
                    finished.append(hyp)
                    continue

                # ── FIX: hard length guard before every decoder call ──────
                # If this hypothesis has already filled the context window,
                # treat it as finished rather than triggering the tensor
                # size mismatch inside model.decoder().
                if len(hyp.token_ids) >= n_text_ctx:
                    hyp.is_finished = True
                    finished.append(hyp)
                    continue
                # ─────────────────────────────────────────────────────────

                # ── Whisper decoder forward pass ──────────────────────────
                token_tensor = torch.tensor(
                    [hyp.token_ids], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    logits = model.decoder(token_tensor, audio_features)
                # logits: (1, seq_len, vocab_size) → last step
                step_logits = logits[0, -1, :]   # (vocab_size,)

                # ── Apply LM logit bias: z'_t = z_t + alpha * LM_bias ────
                step_logits = self.bias_logits(
                    step_logits, hyp.token_ids, step
                )

                # ── Temperature scaling ───────────────────────────────────
                if self.temperature > 0:
                    step_logits = step_logits / self.temperature

                # ── Convert to log-probabilities ──────────────────────────
                log_probs = F.log_softmax(step_logits, dim=-1)

                # ── Expand beam ───────────────────────────────────────────
                top_lp, top_ids = log_probs.topk(self.beam_size)

                for lp, tok_id in zip(top_lp.tolist(), top_ids.tolist()):
                    new_hyp = hyp.clone()
                    new_hyp.token_ids.append(tok_id)
                    new_hyp.asr_score += lp

                    # Decode word for LM scoring and domain tracking
                    try:
                        word = tokenizer.decode([tok_id]).strip().lower()
                    except Exception:
                        word = ""

                    # LM score for this token
                    context_words = self._token_ids_to_words_fast(new_hyp.token_ids, tokenizer)
                    if word and word in self.hard_vocab:
                        lm_lp = max(self.lm.log_prob(word, context_words[-2:]),
                                    self.min_lm_logprob)
                        new_hyp.lm_score += self.alpha * lm_lp
                        new_hyp.domain_hits.add(word)

                    if word:
                        new_hyp.words.append(word)

                    # Coverage penalty
                    cov = self._coverage_penalty(new_hyp, required_words)

                    # Combined score: ASR + alpha*LM - beta*coverage
                    new_hyp.combined_score = new_hyp.asr_score + new_hyp.lm_score - cov

                    if tok_id == eot_id:
                        new_hyp.is_finished = True
                        finished.append(new_hyp)
                    else:
                        all_candidates.append(new_hyp)

            # Keep top beam_size unfinished hypotheses
            all_candidates.sort(key=lambda h: h.combined_score, reverse=True)
            beams = all_candidates[:self.beam_size]

        # If nothing finished, take best unfinished beam
        all_hyps = finished + beams
        if not all_hyps:
            return "", 0.0

        best = max(all_hyps, key=lambda h: h.score_per_token())

        # Decode final text from token IDs (strip prompt tokens)
        output_ids = best.token_ids[len(init_tokens):]
        # Remove EOT if present
        if output_ids and output_ids[-1] == eot_id:
            output_ids = output_ids[:-1]

        text = tokenizer.decode(output_ids).strip()
        n_out = max(len(output_ids), 1)
        avg_lp = best.asr_score / n_out

        log.debug(
            "Beam search done | best_score=%.4f | domain_hits=%s | text=%r",
            best.combined_score, best.domain_hits, text[:60],
        )
        return text, float(avg_lp)

    # ─────────────────────────────────────────────────────────────────────
    # generate() — called by Transcriber
    # ─────────────────────────────────────────────────────────────────────

    def generate(
        self,
        model,
        mel:             torch.Tensor,
        language:        str,
        word_timestamps: bool,
        seg_offset:      float,
        seg_id:          int,
        lang:            str,
        required_words:  Optional[Set[str]] = None,
        seg_duration:    Optional[float] = None,   # FIX: actual clip duration in seconds
    ) -> Tuple[Optional[str], List, float]:
        """
        Entry point called by Transcriber._apply_constrained_decoding().

        Returns
        -------
        (text, word_tokens, avg_log_prob)
        text = None if generation fails or is silence

        Parameters
        ----------
        seg_duration : actual duration of the audio clip in seconds.
                       When None, estimated from the mel length.
                       IMPORTANT: mel is padded to 30 s by pad_or_trim so
                       mel.shape[-1] / 100 ≠ actual segment duration.
                       Always pass this from the transcriber.
        """
        from stt.transcriber import WordToken

        # Resolve device robustly — handles both CPU and CUDA models
        try:
            device = next(model.parameters()).device.type
        except StopIteration:
            device = "cpu"

        try:
            text, avg_lp = self.beam_search(
                model          = model,
                mel            = mel,
                language       = language,
                required_words = required_words or set(),
                device         = device,
            )
        except Exception as exc:
            log.warning("ConstrainedDecoder.generate failed: %s", exc)
            return None, [], 0.0

        if not text:
            return None, [], 0.0

        # Build word-level tokens (no sub-word timestamps from beam search)
        words_list = text.split()
        n = max(len(words_list), 1)

        # FIX: mel is padded to N_FRAMES (30 s * 100 fps = 3000 frames) by
        # whisper.pad_or_trim, so mel.shape[-1] / 100 always gives ~30 s
        # regardless of actual clip length.  Use the caller-supplied duration
        # when available; fall back to the un-padded estimate only if absent.
        if seg_duration is not None and seg_duration > 0:
            dur = seg_duration
        else:
            # Whisper mel hop = 10 ms → 100 fps, but cap at 30 s sanity limit
            dur = min(mel.shape[-1] / 100.0, 30.0)

        step = dur / n

        word_tokens = [
            WordToken(
                word       = w,
                start      = seg_offset + i * step,
                end        = seg_offset + (i + 1) * step,
                lang       = lang,
                confidence = float(np.exp(avg_lp)),
                segment_id = seg_id,
            )
            for i, w in enumerate(words_list)
        ]

        return text, word_tokens, float(avg_lp)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _token_ids_to_words(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to word strings using stored tokenizer."""
        if self._tokenizer is None:
            return []
        try:
            text = self._tokenizer.decode(token_ids)
            return text.strip().lower().split()
        except Exception:
            return []

    def _token_ids_to_words_fast(self, token_ids: List[int], tokenizer) -> List[str]:
        """Decode token IDs to word strings (accepts tokenizer arg directly)."""
        try:
            text = tokenizer.decode(token_ids)
            return text.strip().lower().split()
        except Exception:
            return []

    def __repr__(self) -> str:
        return (
            f"ConstrainedDecoder("
            f"alpha={self.alpha}, beta={self.beta}, "
            f"beam={self.beam_size}, "
            f"vocab={len(self.hard_vocab)} words, "
            f"lm={self.lm!r})"
        )

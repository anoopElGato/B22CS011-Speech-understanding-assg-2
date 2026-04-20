"""
stt/ngram_lm.py
================
N-gram Language Model for constrained ASR decoding.

Trains on a custom corpus of domain-specific words (e.g. a technical
syllabus) so that the downstream decoder can bias Whisper toward
in-domain vocabulary.

Architecture
------------
  - NLTK MLE / Laplace / KneserNey smoothed N-gram model
  - Optionally wraps kenlm for faster inference (if installed)
  - Exposes log_prob(word, context) and vocab for logit biasing

Mathematical Background
-----------------------
  Standard N-gram probability:

      P(w_t | w_{t-n+1}, ..., w_{t-1})
          = C(w_{t-n+1}, ..., w_t) / C(w_{t-n+1}, ..., w_{t-1})

  With Kneser-Ney smoothing:
      P_KN(w | h) = max(C(h,w) - d, 0) / C(h)
                  + lambda(h) * P_KN_lower(w)

  where d is the discount parameter and lambda(h) is a normalisation
  constant ensuring probabilities sum to 1.

Usage
-----
    lm = NgramLM(n=3, smoothing="kneser_ney")
    lm.train(corpus_path="data/syllabus_corpus.txt")
    prob = lm.log_prob("transformer", context=["the", "attention"])
    # or load pre-built corpus from list of strings
    lm.train_from_sentences(["transformers use attention", ...])
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Default technical syllabus corpus — domain words for logit biasing
DEFAULT_CORPUS = """
प्रधानमंत्री प्रधानमंत्री जी नरेंद्र मोदी भारत सरकार केंद्र सरकार head of the government
देश देशवासी देशवासियों सेवा जनसेवा राष्ट्र राष्ट्रीय भारतवासी हिंदुस्तान
जिम्मेवारी जिम्मेदारी परिश्रम प्रयास प्रेरणा aspiration आवश्यकता ऊर्जा motivation motivate
चुनाव चुनावी अभियान वादा विकास जनहित जनकल्याण लोकहित सुशासन governance administration
grassroot grassroots first hand information district जिलों रात्रि मुकाम भ्रमण अनुभव
politician politician leader नेता फैसला निर्णय decision making process conviction reaction
governance की दृष्टि baggage without baggage public life leadership responsibility
मौज मस्ती पद कर्तव्य dedication devotion devote hard work तपस्या
service aspiration inspiration public service national service
engineer engineering गणित mathematician mathematics mathematical ideas
श्रीनिवास रामानुजन ramanujan science spirituality spiritual scientific advance minds
देवी पूजा ideas knowledge information processing evolve sources open mind
महात्मा गांधी गांधी गरीब का चेहरा सामान्य मानव निर्णय मंत्र
officer officers अफसर अफसरों brief information channel devil advocate विद्यार्थी भाव
मंथन अमृत conviction प्रक्रिया पूछता हूं समझने का प्रयास
corona covid lockdown economy economists inflation Nobel Prize expert opinion
खजाना नोटों छापो गरीब भूखा सामाजिक तनाव रोजमर्रा की आवश्यकताएं
world economy global economy बड़ी economy तेज गति से प्रगति
policy policymaking governance reform implementation administration governance model
गुजरात भारत सरकार district development welfare delivery public delivery system
1 4 billion people 1 4 billion लोगों की सेवा billion aspiration mood
14 में चुनाव 24 साल head of government लंबे कालखंड
bad इरादे bad irade मेरे लिए मैं कुछ नहीं करूंगा
मैं देशवासियों को वादा करता हूं परिश्रम करने में पीछे नहीं रहूंगा
मेरे प्रयास में कमी नहीं रहेगी मेरे परिश्रम में कमी नहीं रहेगी
मैं बद इरादे से कोई काम नहीं करूंगा
मेरे लिए मैं कुछ नहीं करूंगा
मेरी जिम्मेवारी मुझे दौड़ाती है
मेरे देश का नुकसान तो नहीं हो रहा है
कोई गरीब का चेहरा देख लो
मैं सामान्य मानव को याद करता हूं
मैं बहुत well connected हूं
मेरे information channel बहुत live हैं
मेरे पास first hand information है
मैं devil advocate बनकर प्रश्न पूछता हूं
मैं विद्यार्थी भाव से पूछता हूं
मुझे knowledge और information का फर्क समझना है
science और spirituality के बीच बहुत बड़ा connect है
mathematical ideas तपस्या से आते हैं
hard work devotion dedication तपस्या
lockdown के समय economy inflation expert opinion
मेरे देश की परिस्थिति मेरे अनुभव मेरे निर्णय
गरीब को भूखा सोने नहीं दूंगा
social tension पैदा नहीं होने दूंगा
immediate after covid inflation crisis
public welfare governance economic stability
प्रधानमंत्री government governance policy decision economy inflation lockdown
रामानुजन science spirituality mathematical ideas engineer गणित
"""


# ─────────────────────────────────────────────────────────────────────────────
# NgramLM
# ─────────────────────────────────────────────────────────────────────────────

class NgramLM:
    """
    N-gram language model with smoothing, built on NLTK.

    Parameters
    ----------
    n         : order of the n-gram model (default 3 = trigram)
    smoothing : "kneser_ney" | "laplace" | "mle"
    vocab_size: maximum vocabulary size (0 = unlimited)
    """

    def __init__(
        self,
        n:          int = 3,
        smoothing:  str = "kneser_ney",
        vocab_size: int = 0,
    ):
        self.n         = n
        self.smoothing = smoothing
        self.vocab_size = vocab_size

        # Raw counts (always built, even if NLTK is used for smoothed model)
        self._unigram_counts: Counter = Counter()
        self._ngram_counts:   Dict    = defaultdict(Counter)
        self._total_tokens:   int     = 0

        # NLTK model (set after train())
        self._nltk_model = None
        self._vocab      = None

        # Word set for fast lookup (set after train())
        self.word_set: set = set()

        log.info("NgramLM init | n=%d | smoothing=%s", n, smoothing)

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def train_from_sentences(self, sentences: List[str]) -> "NgramLM":
        """
        Train the language model from a list of plain-text sentences.

        Parameters
        ----------
        sentences : list of tokenised or raw strings

        Returns
        -------
        self
        """
        try:
            import nltk
            from nltk.lm import MLE, Laplace
            from nltk.lm.preprocessing import (
                padded_everygram_pipeline, pad_both_ends
            )
            try:
                from nltk.lm import KneserNeyInterpolated
            except ImportError:
                KneserNeyInterpolated = Laplace   # graceful fallback

            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        except ImportError:
            raise ImportError("Run: pip install nltk")

        # Tokenise
        tokenised = [self._tokenise(s) for s in sentences if s.strip()]

        # Build count-based vocab
        all_tokens = [t for sent in tokenised for t in sent]
        self._unigram_counts = Counter(all_tokens)
        self._total_tokens   = len(all_tokens)

        # Build n-gram counts manually (used for fast log_prob fallback)
        self._build_raw_counts(tokenised)

        # Vocabulary — optionally limit to top-k
        if self.vocab_size > 0:
            top_words = {w for w, _ in self._unigram_counts.most_common(self.vocab_size)}
        else:
            top_words = set(self._unigram_counts.keys())

        self.word_set = top_words

        # NLTK padded everygram pipeline
        train_data, padded_vocab = padded_everygram_pipeline(self.n, tokenised)

        # Instantiate model based on smoothing choice
        if self.smoothing == "kneser_ney":
            self._nltk_model = KneserNeyInterpolated(self.n)
        elif self.smoothing == "laplace":
            self._nltk_model = Laplace(self.n)
        else:
            self._nltk_model = MLE(self.n)

        self._nltk_model.fit(train_data, padded_vocab)
        self._vocab = self._nltk_model.vocab

        log.info(
            "NgramLM trained | tokens=%d | vocab=%d | n=%d | smoothing=%s",
            self._total_tokens, len(self.word_set), self.n, self.smoothing,
        )
        return self

    def train(self, corpus_path: str) -> "NgramLM":
        """
        Train from a plain-text file (one sentence per line, or free text).
        """
        text = Path(corpus_path).read_text(encoding="utf-8")
        sentences = self._split_sentences(text)
        return self.train_from_sentences(sentences)

    def train_from_default_corpus(self) -> "NgramLM":
        """
        Train on the built-in technical/domain corpus (no file needed).
        Useful for getting started without any external data.
        """
        sentences = self._split_sentences(DEFAULT_CORPUS)
        # Augment with individual term sentences for better unigram coverage
        extra = [line.strip() for line in DEFAULT_CORPUS.splitlines() if line.strip()]
        return self.train_from_sentences(sentences + extra)

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def log_prob(
        self,
        word:    str,
        context: Optional[List[str]] = None,
    ) -> float:
        """
        Return log P(word | context).

        Uses the NLTK smoothed model when available; falls back to
        Laplace-smoothed raw counts otherwise.

        Parameters
        ----------
        word    : target word
        context : preceding words (most recent last), length ≤ n-1
                  None or [] → unigram probability

        Returns
        -------
        float  log-probability (base-e), in range (-∞, 0]
        """
        word = word.lower().strip()

        # ── NLTK model path ──────────────────────────────────────────────
        if self._nltk_model is not None:
            ctx = tuple((context or [])[-self.n + 1:])
            try:
                p = self._nltk_model.score(word, ctx)
                p = max(p, 1e-12)
                return math.log(p)
            except Exception:
                pass  # fall through to raw counts

        # ── Raw-count Laplace fallback ────────────────────────────────────
        return self._laplace_log_prob(word, context)

    def _laplace_log_prob(
        self,
        word: str,
        context: Optional[List[str]] = None,
    ) -> float:
        """Laplace-smoothed n-gram log-prob from raw counts."""
        V = max(len(self._unigram_counts), 1)

        if not context:
            count_w = self._unigram_counts.get(word, 0)
            return math.log((count_w + 1) / (self._total_tokens + V))

        ctx = tuple(context[-self.n + 1:])
        count_ctx_w = self._ngram_counts.get(ctx, Counter()).get(word, 0)
        count_ctx   = sum(self._ngram_counts.get(ctx, Counter()).values())
        return math.log((count_ctx_w + 1) / (count_ctx + V))

    def perplexity(self, sentences: List[str]) -> float:
        """
        Compute model perplexity on held-out sentences.
        Lower perplexity = better model for this domain.
        """
        total_log_prob = 0.0
        total_tokens   = 0
        for sent in sentences:
            tokens = self._tokenise(sent)
            for i, word in enumerate(tokens):
                ctx = tokens[max(0, i - self.n + 1):i]
                total_log_prob += self.log_prob(word, ctx)
                total_tokens   += 1
        if total_tokens == 0:
            return float("inf")
        avg_log_prob = total_log_prob / total_tokens
        return math.exp(-avg_log_prob)

    def top_k_words(self, k: int = 50) -> List[Tuple[str, int]]:
        """Return the k most frequent words in the training corpus."""
        return self._unigram_counts.most_common(k)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase, keep Latin + Devanagari tokens, strip other punctuation."""
        text  = text.lower()
        text  = re.sub(r"[^\u0900-\u097fa-z0-9'\- ]", " ", text)
        return text.split()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on newlines, periods, and Devanagari danda."""
        sentences = re.split(r"[.\n।]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_raw_counts(self, tokenised: List[List[str]]) -> None:
        """Accumulate n-gram counts for the fallback log_prob path."""
        for tokens in tokenised:
            for i in range(len(tokens)):
                for order in range(1, self.n + 1):
                    if i >= order - 1:
                        ctx  = tuple(tokens[i - order + 1:i])
                        word = tokens[i]
                        self._ngram_counts[ctx][word] += 1

    def __repr__(self) -> str:
        trained = f"vocab={len(self.word_set)}" if self.word_set else "untrained"
        return f"NgramLM(n={self.n}, smoothing={self.smoothing!r}, {trained})"

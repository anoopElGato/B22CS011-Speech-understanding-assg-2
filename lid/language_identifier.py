"""
lid/language_identifier.py
============================
Frame-level Language Identification for code-switched Hinglish audio.

Uses the pretrained SpeechBrain VoxLingua107 ECAPA-TDNN model — NO training
from scratch.  All custom logic (windowing, smoothing, segmentation,
timestamp refinement, oscillation suppression) is implemented here.

Model:  speechbrain/lang-id-voxlingua107-ecapa
        — 107-language classifier trained on VoxLingua107
        — backbone: ECAPA-TDNN (x-vector style)
        — output: softmax distribution over 107 language labels

Pipeline
--------
  WAV file / waveform
      │
      ▼
  preprocess_audio()     — resample → 16 kHz, mono, normalise amplitude
      │
      ▼
  frame_audio()          — sliding windows (0.5 s window, 0.1 s hop)
      │
      ▼
  predict_frames()       — SpeechBrain inference on every window
                           extract P(hi) and P(en) from 107-class softmax
      │
      ▼
  smooth_predictions()   — Gaussian-weighted moving average over probabilities
                           + median filter on hard labels
      │
      ▼
  merge_segments()       — run-length encode frame labels → raw segments
      │
      ▼
  refine_timestamps()    — drop segments < 200 ms, absorb into neighbours
                           suppress oscillation runs
      │
      ▼
  List[Dict]  →  [{"start": 0.0, "end": 2.3, "lang": "en"}, ...]

Usage
-----
    from lid.language_identifier import LIDModule

    lid = LIDModule()
    segments = lid.predict("audio/hinglish_sample.wav")
    for seg in segments:
        print(f"{seg['start']:.2f}s - {seg['end']:.2f}s : {seg['lang']}")

    # Optional: visualise timeline
    lid.plot_timeline(segments, total_duration=10.5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR       = 16_000   # SpeechBrain model expects 16 kHz
WINDOW_SEC      = 0.5      # sliding window duration
HOP_SEC         = 0.1      # hop between consecutive windows
MIN_SEG_SEC     = 0.2      # minimum segment duration before absorption
SMOOTH_WINDOW   = 7        # Gaussian smoothing kernel length (frames)
MEDIAN_KERNEL   = 5        # median filter kernel size (frames, must be odd)

# VoxLingua107 uses BCP-47 codes — map only what we care about
VOXLINGUA_TO_LANG: Dict[str, str] = {
    "hi": "hi",   # Hindi
    "en": "en",   # English
}

# ─────────────────────────────────────────────────────────────────────────────
# Frame-level result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FramePrediction:
    """Per-window inference output (raw + smoothed)."""
    frame_idx:      int
    start_sec:      float
    end_sec:        float
    prob_hi:        float          # P(Hindi)  before smoothing
    prob_en:        float          # P(English) before smoothing
    raw_lang:       str            # argmax before smoothing
    smooth_prob_hi: float = 0.0
    smooth_prob_en: float = 0.0
    smooth_lang:    str   = ""


@dataclass
class FGSMAttackResult:
    """Container for a targeted FGSM adversarial example."""

    perturbed_audio: np.ndarray
    epsilon_used: float
    snr_db: float
    original_prediction: str
    perturbed_prediction: str
    original_prob_hi: float
    original_prob_en: float
    perturbed_prob_hi: float
    perturbed_prob_en: float


# ─────────────────────────────────────────────────────────────────────────────
# LIDModule
# ─────────────────────────────────────────────────────────────────────────────

class LIDModule:
    """
    Frame-level Language Identifier for Hinglish (Hindi / English) audio.

    Wraps the pretrained SpeechBrain VoxLingua107 ECAPA model and adds
    custom sliding-window inference, probabilistic smoothing, segment
    merging, and timestamp refinement.

    Parameters
    ----------
    model_name    : HuggingFace / SpeechBrain model source string
    savedir       : local cache directory for the downloaded model
    window_sec    : sliding window duration in seconds   (default 0.5)
    hop_sec       : hop between consecutive windows      (default 0.1)
    smooth_window : Gaussian smoothing kernel length in frames
    median_kernel : median filter kernel size in frames  (must be odd)
    min_seg_sec   : minimum segment duration; shorter ones get absorbed
    device        : 'cpu', 'cuda', or None (auto-detect)
    target_langs  : mapping {VoxLingua107_code: output_label}
    """

    def __init__(
        self,
        model_name:    str = "speechbrain/lang-id-voxlingua107-ecapa",
        savedir:       str = "pretrained_models/lang_id",
        window_sec:    float = WINDOW_SEC,
        hop_sec:       float = HOP_SEC,
        smooth_window: int   = SMOOTH_WINDOW,
        median_kernel: int   = MEDIAN_KERNEL,
        min_seg_sec:   float = MIN_SEG_SEC,
        device:        Optional[str] = None,
        target_langs:  Optional[Dict[str, str]] = None,
    ):
        self.model_name   = model_name
        self.savedir      = savedir
        self.window_sec   = window_sec
        self.hop_sec      = hop_sec
        self.smooth_window = smooth_window
        # Median kernel must be odd
        self.median_kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        self.min_seg_sec  = min_seg_sec
        self.target_langs = target_langs or VOXLINGUA_TO_LANG

        # Auto-select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-initialised by load_model()
        self._model:      Optional[object]  = None
        self._label_list: List[str]         = []
        self._backend:    str               = "speechbrain"

        log.info(
            "LIDModule ready | window=%.1fs | hop=%.1fs | "
            "smooth=%d frames | min_seg=%.2fs | device=%s",
            window_sec, hop_sec, smooth_window, min_seg_sec, self.device,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_model(self) -> "LIDModule":
        """
        Download (first run only) and load the VoxLingua107 ECAPA model.

        The model is cached in self.savedir so subsequent loads are instant.
        Returns self to allow method chaining.
        """
        if self._model is not None:
            return self   # already loaded

        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "SpeechBrain is not installed.\n"
                "Run: pip install speechbrain>=1.0"
            ) from exc

        log.info("Loading pretrained model: %s", self.model_name)

        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
            self._model.eval()
        except Exception as exc:
            log.warning(
                "SpeechBrain LID model could not be loaded; falling back to Whisper language detection. (%s)",
                exc,
            )
            try:
                import whisper
            except ImportError as inner_exc:
                raise RuntimeError(
                    "SpeechBrain LID failed to load and Whisper fallback is unavailable. "
                    "Install openai-whisper or fix the SpeechBrain/k2 environment."
                ) from inner_exc

            self._model = whisper.load_model("tiny", device=self.device)
            self._label_list = ["hi", "en"]
            self._backend = "whisper"
            log.info("Whisper fallback LID model loaded.")
            return self

        # Extract the ordered label list from the model's label encoder so
        # we can map class index → language code without hard-coding order.
        self._label_list = self._extract_label_list()
        self._backend = "speechbrain"

        log.info(
            "Model loaded | %d language classes | "
            "tracking: %s",
            len(self._label_list),
            list(self.target_langs.keys()),
        )
        return self

    def _extract_label_list(self) -> List[str]:
        """
        Pull the ordered language label list out of the SpeechBrain model.
        Falls back to the known VoxLingua107 alphabetical ordering.
        """
        try:
            encoder = self._model.hparams.label_encoder
            # decode_ndim(i) returns the string label for class index i
            return [encoder.decode_ndim(i) for i in range(len(encoder.lab2ind))]
        except Exception:
            log.warning(
                "Could not extract label list from model internals; "
                "using built-in VoxLingua107 ordering as fallback."
            )
            return self._voxlingua107_fallback_labels()

    @staticmethod
    def _voxlingua107_fallback_labels() -> List[str]:
        """
        Alphabetically sorted BCP-47 codes matching the VoxLingua107 release.
        This ordering mirrors the SpeechBrain label encoder for this model.
        """
        return [
            "ab","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs",
            "ca","ceb","cs","cy","da","de","el","en","eo","es","et","eu","fa",
            "fi","fo","fr","gl","gn","gu","ha","haw","hi","ht","hu","hy","ia",
            "id","is","it","iw","ja","jw","ka","kk","km","kn","ko","la","lb",
            "ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my",
            "ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","si",
            "sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th",
            "tk","tl","tr","tt","uk","ur","uz","vi","vo","war","yi","yo","zh",
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Audio preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def preprocess_audio(
        self,
        source:    "str | np.ndarray | torch.Tensor",
        source_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load, convert to mono, resample to 16 kHz, and peak-normalise.

        Accepts
        -------
        - file path  (str or pathlib.Path)
        - NumPy array  shape (samples,) or (channels, samples)
        - Torch tensor shape (samples,) or (channels, samples)

        Returns
        -------
        waveform : (1, samples)  float32 Torch tensor at TARGET_SR
        sr       : int  always equal to TARGET_SR after resampling
        """
        # ── Load from various source types ────────────────────────────────
        if isinstance(source, (str, Path)):
            waveform, sr = torchaudio.load(str(source))  # (C, T)
        elif isinstance(source, np.ndarray):
            waveform = torch.from_numpy(
                source[np.newaxis] if source.ndim == 1 else source
            ).float()
            sr = source_sr or TARGET_SR
        elif isinstance(source, torch.Tensor):
            waveform = source.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = source_sr or TARGET_SR
        else:
            raise TypeError(f"Unsupported audio source type: {type(source)}")

        # ── Convert to mono ───────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ── Resample to 16 kHz ────────────────────────────────────────────
        if sr != TARGET_SR:
            log.info("Resampling %d Hz → %d Hz", sr, TARGET_SR)
            waveform = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)

        # ── Peak normalisation (leave 10 % headroom) ──────────────────────
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.9

        waveform = waveform.contiguous().float()
        duration = waveform.shape[1] / TARGET_SR
        log.info("Audio preprocessed: %.2f s  shape=%s", duration, tuple(waveform.shape))
        return waveform, TARGET_SR

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Sliding-window framing
    # ─────────────────────────────────────────────────────────────────────────

    def frame_audio(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """
        Slice the waveform into overlapping fixed-length windows.

        Window size : self.window_sec  (default 0.5 s = 8 000 samples @ 16 kHz)
        Hop size    : self.hop_sec     (default 0.1 s = 1 600 samples @ 16 kHz)

        Tail handling
        -------------
        - Tail window >= 50 % full → zero-pad to full window size
        - Tail window <  50 % full → discarded (too short to classify)

        Returns
        -------
        windows    : List of (1, window_samples) float32 tensors
        timestamps : List of (start_sec, end_sec) tuples
        """
        total_samples  = waveform.shape[1]
        win_samples    = int(self.window_sec * TARGET_SR)
        hop_samples    = int(self.hop_sec    * TARGET_SR)

        windows:    List[torch.Tensor]          = []
        timestamps: List[Tuple[float, float]]   = []

        start = 0
        while start < total_samples:
            end  = start + win_samples
            clip = waveform[:, start:min(end, total_samples)]  # (1, t)

            if clip.shape[1] < win_samples:
                fraction = clip.shape[1] / win_samples
                if fraction < 0.5:
                    break   # tail too short — discard
                # Zero-pad right side to fill the window
                pad  = torch.zeros(1, win_samples - clip.shape[1])
                clip = torch.cat([clip, pad], dim=1)

            windows.append(clip)
            timestamps.append((
                start / TARGET_SR,
                min(end, total_samples) / TARGET_SR,
            ))
            start += hop_samples

        log.info(
            "Framed into %d windows (window=%.2fs, hop=%.2fs)",
            len(windows), self.window_sec, self.hop_sec,
        )
        return windows, timestamps

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Per-frame inference
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_frames(
        self,
        windows:    List[torch.Tensor],
        timestamps: List[Tuple[float, float]],
    ) -> List[FramePrediction]:
        """
        Run the SpeechBrain classifier on every window and extract
        per-frame Hindi and English probabilities.

        Processing per window
        ---------------------
        1. Forward pass through ECAPA-TDNN → utterance-level embedding
        2. Linear classifier → 107-dim vector (log-softmax or softmax)
        3. Convert to probability (exp if log-softmax)
        4. Locate hi / en indices in the label list
        5. Renormalise P(hi) and P(en) so they sum to 1 (binary decision)

        Returns
        -------
        List[FramePrediction], one entry per window, in chronological order.
        """
        if self._model is None:
            self.load_model()

        n        = len(windows)
        results: List[FramePrediction] = []

        for i, (win, (t_start, t_end)) in enumerate(zip(windows, timestamps)):

            # SpeechBrain classify_batch() expects (batch=1, time) tensor
            wav_in = win.to(self.device)              # (1, win_samples)

            if self._backend != "speechbrain":
                prob_hi, prob_en = self._predict_window_probs_whisper(wav_in)
                raw_lang = "hi" if prob_hi >= prob_en else "en"

                results.append(FramePrediction(
                    frame_idx = i,
                    start_sec = t_start,
                    end_sec   = t_end,
                    prob_hi   = prob_hi,
                    prob_en   = prob_en,
                    raw_lang  = raw_lang,
                ))

                if (i + 1) % 100 == 0 or i == n - 1:
                    log.info("  Inference: %d / %d windows", i + 1, n)
                continue

            if self._backend == "speechbrain":
                lang_idx = self._build_lang_index()
                out_probs, _score, _index, _label = self._model.classify_batch(wav_in)

            # Convert to numpy probability vector (handle log-softmax output)
            probs = out_probs[0].float().cpu().numpy()
            if probs.min() < 0:          # log-softmax → softmax
                probs = np.exp(probs)
            probs = np.clip(probs, 0.0, 1.0)
            probs /= probs.sum() + 1e-12  # renormalise to sum-to-1

            # Extract and renormalise to binary (hi vs en)
            prob_hi = float(probs[lang_idx.get("hi", 0)])
            prob_en = float(probs[lang_idx.get("en", 0)])
            total   = prob_hi + prob_en + 1e-12
            prob_hi /= total
            prob_en /= total

            raw_lang = "hi" if prob_hi >= prob_en else "en"

            results.append(FramePrediction(
                frame_idx = i,
                start_sec = t_start,
                end_sec   = t_end,
                prob_hi   = prob_hi,
                prob_en   = prob_en,
                raw_lang  = raw_lang,
            ))

            if (i + 1) % 20 == 0 or i == n - 1:
                log.info("  Inference: %d / %d windows", i + 1, n)

        return results

    def _build_lang_index(self) -> Dict[str, int]:
        """
        Map target language codes → class index in the 107-dim softmax.

        Handles hyphenated codes like "zh-cn" by matching the prefix only.
        Raises RuntimeError if neither hi nor en is found.
        """
        idx: Dict[str, int] = {}
        for i, label in enumerate(self._label_list):
            code = label.strip().lower().split("-")[0]
            for target in self.target_langs:
                if code == target and target not in idx:
                    idx[target] = i

        missing = [t for t in self.target_langs if t not in idx]
        if missing:
            raise RuntimeError(
                f"Target language(s) {missing} not found in model label list.\n"
                f"First 10 labels: {self._label_list[:10]}"
            )
        return idx

    @torch.no_grad()
    def _predict_window_probs_whisper(self, window: torch.Tensor) -> Tuple[float, float]:
        """Fallback Hindi/English probabilities using Whisper language detection."""
        import whisper

        wav = window.squeeze(0).detach().float().cpu()
        wav = whisper.pad_or_trim(wav)
        mel = whisper.log_mel_spectrogram(wav).to(self.device)
        _, lang_probs = self._model.detect_language(mel)

        prob_hi = float(lang_probs.get("hi", 0.0))
        prob_en = float(lang_probs.get("en", 0.0))
        total = prob_hi + prob_en + 1e-12
        return prob_hi / total, prob_en / total

    def _binary_probs_from_window(self, window: torch.Tensor) -> torch.Tensor:
        """Return differentiable [P(hi), P(en)] for a single analysis window."""
        if self._model is None:
            self.load_model()
        if self._backend != "speechbrain":
            raise RuntimeError("Differentiable binary probabilities require the SpeechBrain LID backend.")

        lang_idx = self._build_lang_index()
        out_probs, _score, _index, _label = self._model.classify_batch(window.to(self.device))

        probs = out_probs[0].float()
        if probs.min().item() < 0:
            probs = torch.exp(probs)
        probs = probs / (probs.sum() + 1e-12)

        prob_hi = probs[lang_idx["hi"]]
        prob_en = probs[lang_idx["en"]]
        binary = torch.stack([prob_hi, prob_en])
        binary = binary / (binary.sum() + 1e-12)
        return binary

    def _predict_binary_probs_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        """Average frame-level Hindi/English probabilities over the utterance."""
        windows, _timestamps = self.frame_audio(waveform)
        if not windows:
            raise ValueError("Audio too short to produce any frames for LID.")

        if self._backend == "speechbrain":
            frame_probs = [self._binary_probs_from_window(window) for window in windows]
        else:
            frame_probs = [
                torch.tensor(
                    self._predict_window_probs_whisper(window),
                    dtype=torch.float32,
                    device=self.device,
                )
                for window in windows
            ]
        utterance_probs = torch.stack(frame_probs, dim=0).mean(dim=0)
        utterance_probs = utterance_probs / (utterance_probs.sum() + 1e-12)
        return utterance_probs

    @torch.no_grad()
    def predict_utterance(
        self,
        audio: "str | np.ndarray | torch.Tensor",
        source_sr: Optional[int] = None,
    ) -> Dict[str, float | str]:
        """Return utterance-level Hindi/English probabilities and prediction."""
        self.load_model()
        waveform, _ = self.preprocess_audio(audio, source_sr)
        probs = self._predict_binary_probs_tensor(waveform)
        prob_hi = float(probs[0].detach().cpu().item())
        prob_en = float(probs[1].detach().cpu().item())
        return {
            "prediction": "hi" if prob_hi >= prob_en else "en",
            "prob_hi": prob_hi,
            "prob_en": prob_en,
        }

    @staticmethod
    def _snr_db(clean: torch.Tensor, perturbed: torch.Tensor) -> float:
        """Compute signal-to-noise ratio in dB."""
        noise = perturbed - clean
        signal_power = torch.mean(clean.pow(2)).item()
        noise_power = torch.mean(noise.pow(2)).item()
        if noise_power <= 1e-20:
            return float("inf")
        return float(10.0 * np.log10((signal_power + 1e-20) / (noise_power + 1e-20)))

    def fgsm_attack(
        self,
        audio: "str | np.ndarray | torch.Tensor",
        source_sr: Optional[int] = None,
        epsilon: float = 0.01,
        min_snr_db: float = 40.0,
        search_steps: int = 10,
    ) -> FGSMAttackResult:
        """
        Generate a targeted FGSM example that tries to flip the utterance label.
        """
        self.load_model()
        if self._backend != "speechbrain":
            raise RuntimeError(
                "FGSM attack requires the SpeechBrain LID backend. "
                "Current backend is Whisper fallback because SpeechBrain could not load."
            )
        clean_waveform, _ = self.preprocess_audio(audio, source_sr)
        clean_waveform = clean_waveform.to(self.device)

        original_probs = self._predict_binary_probs_tensor(clean_waveform)
        original_label_idx = int(torch.argmax(original_probs).item())
        target_label_idx = 1 - original_label_idx
        original_prediction = "hi" if original_label_idx == 0 else "en"

        adv_source = clean_waveform.clone().detach().requires_grad_(True)
        adv_probs = self._predict_binary_probs_tensor(adv_source)
        loss = -torch.log(adv_probs[target_label_idx] + 1e-12)

        if adv_source.grad is not None:
            adv_source.grad.zero_()
        if hasattr(self._model, "zero_grad"):
            self._model.zero_grad()
        loss.backward()

        grad_sign = adv_source.grad.sign()
        signal_power = torch.mean(clean_waveform.pow(2)).item()
        snr_limited_epsilon = float(np.sqrt(signal_power / (10.0 ** (min_snr_db / 10.0))))
        max_epsilon = max(0.0, min(float(epsilon), snr_limited_epsilon * 0.999))

        candidate_epsilons = np.linspace(
            max_epsilon / max(search_steps, 1),
            max_epsilon,
            max(search_steps, 1),
        )

        best_waveform = clean_waveform.clone().detach()
        best_probs = original_probs.detach()
        epsilon_used = 0.0
        perturbed_prediction = original_prediction

        for candidate_epsilon in candidate_epsilons:
            perturbed = clean_waveform - float(candidate_epsilon) * grad_sign
            perturbed = torch.clamp(perturbed, -1.0, 1.0)

            with torch.no_grad():
                perturbed_probs = self._predict_binary_probs_tensor(perturbed)

            perturbed_label_idx = int(torch.argmax(perturbed_probs).item())
            best_waveform = perturbed.detach()
            best_probs = perturbed_probs.detach()
            epsilon_used = float(candidate_epsilon)
            perturbed_prediction = "hi" if perturbed_label_idx == 0 else "en"

            if perturbed_label_idx == target_label_idx:
                break

        snr_db = self._snr_db(clean_waveform.detach(), best_waveform)
        if perturbed_prediction == original_prediction:
            raise RuntimeError(
                "FGSM attack could not flip the LID prediction within the allowed epsilon/SNR budget."
            )
        if snr_db < min_snr_db:
            raise RuntimeError(
                f"FGSM attack violated SNR constraint: {snr_db:.2f} dB < {min_snr_db:.2f} dB."
            )

        return FGSMAttackResult(
            perturbed_audio=best_waveform.squeeze(0).detach().cpu().numpy().astype(np.float32),
            epsilon_used=epsilon_used,
            snr_db=snr_db,
            original_prediction=original_prediction,
            perturbed_prediction=perturbed_prediction,
            original_prob_hi=float(original_probs[0].detach().cpu().item()),
            original_prob_en=float(original_probs[1].detach().cpu().item()),
            perturbed_prob_hi=float(best_probs[0].detach().cpu().item()),
            perturbed_prob_en=float(best_probs[1].detach().cpu().item()),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Temporal smoothing
    # ─────────────────────────────────────────────────────────────────────────

    def smooth_predictions(
        self,
        predictions: List[FramePrediction],
    ) -> List[FramePrediction]:
        """
        Two-pass smoothing on frame predictions.

        Pass 1 — Gaussian moving average on probabilities
        ---------------------------------------------------
        Applies a 1-D Gaussian convolution to P(hi) and P(en) separately,
        then renormalises.  Frames closer to the kernel centre receive higher
        weight, creating soft, bell-shaped blending at language boundaries.
        Uses reflect-padding to avoid edge artefacts.

        Pass 2 — Median filter on hard labels
        ----------------------------------------
        After re-argmaxing the smoothed probabilities, a median filter is
        applied to the resulting binary label sequence.  This eliminates
        isolated single-frame flips that the Gaussian pass didn't fully
        remove (because two adjacent opposite-language frames can average
        to ~0.5 and flip back under a low threshold).

        Together the two passes give:
          - Smooth probability transitions (Pass 1)
          - Crisp boundary placement without flickering (Pass 2)

        Modifies each FramePrediction in-place.
        Returns the same list.
        """
        n = len(predictions)
        if n == 0:
            return predictions

        prob_hi = np.array([p.prob_hi for p in predictions])
        prob_en = np.array([p.prob_en for p in predictions])

        # ── Pass 1: Gaussian moving average ──────────────────────────────
        # Build normalised Gaussian kernel of length smooth_window
        klen  = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        sigma = (klen // 2) / 2.0 or 1.0
        x      = np.arange(klen) - klen // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        pad = klen // 2
        # reflect-pad both probability arrays so border frames aren't biased
        hi_conv = np.convolve(np.pad(prob_hi, pad, mode="reflect"), kernel, mode="valid")[:n]
        en_conv = np.convolve(np.pad(prob_en, pad, mode="reflect"), kernel, mode="valid")[:n]

        # Renormalise after convolution
        total  = hi_conv + en_conv + 1e-12
        hi_sm  = hi_conv / total
        en_sm  = en_conv / total

        # ── Pass 2: Median filter on argmax labels ────────────────────────
        # 0 = hi, 1 = en
        hard_labels = (en_sm > hi_sm).astype(np.int8)

        from scipy.ndimage import median_filter
        hard_labels = median_filter(
            hard_labels, size=self.median_kernel, mode="reflect"
        )

        # ── Write smoothed values back into each FramePrediction ─────────
        for i, pred in enumerate(predictions):
            pred.smooth_prob_hi = float(hi_sm[i])
            pred.smooth_prob_en = float(en_sm[i])
            pred.smooth_lang    = "en" if hard_labels[i] == 1 else "hi"

        hi_frac = float(hard_labels.mean())
        log.info(
            "Smoothing done | Gaussian kernel=%d | Median kernel=%d | "
            "EN fraction=%.1f%%",
            klen, self.median_kernel, (1 - hi_frac) * 100,
        )
        return predictions

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Merge consecutive frames → raw segments
    # ─────────────────────────────────────────────────────────────────────────

    def merge_segments(
        self,
        predictions: List[FramePrediction],
    ) -> List[Dict]:
        """
        Run-length encode the smoothed per-frame language labels into
        contiguous time segments.

        Adjacent frames with the same smooth_lang are collapsed into one
        segment spanning from the first frame's start_sec to the last
        frame's end_sec.

        Returns
        -------
        List of segment dicts:
            { "start": float, "end": float, "lang": str, "_n_frames": int }

        The "_n_frames" key is used internally by refine_timestamps() and
        is stripped from the final public output.
        """
        if not predictions:
            return []

        segments: List[Dict] = []
        cur_lang  = predictions[0].smooth_lang
        seg_start = predictions[0].start_sec
        n_frames  = 1

        for pred in predictions[1:]:
            if pred.smooth_lang == cur_lang:
                n_frames += 1
            else:
                segments.append({
                    "start":     seg_start,
                    "end":       pred.start_sec,
                    "lang":      cur_lang,
                    "_n_frames": n_frames,
                })
                cur_lang  = pred.smooth_lang
                seg_start = pred.start_sec
                n_frames  = 1

        # Append the final run
        segments.append({
            "start":     seg_start,
            "end":       predictions[-1].end_sec,
            "lang":      cur_lang,
            "_n_frames": n_frames,
        })

        log.info("Merged into %d raw segments before refinement", len(segments))
        return segments

    # ─────────────────────────────────────────────────────────────────────────
    # 7 & 8. Timestamp refinement + oscillation suppression
    # ─────────────────────────────────────────────────────────────────────────

    def refine_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Two-pass cleanup that guarantees:
          - All output segments have duration >= min_seg_sec
          - No consecutive segments share the same language
          - No temporal gaps between adjacent segments

        Pass A — Short-segment absorption
        -----------------------------------
        Iterates until stable.  On each pass, any segment shorter than
        min_seg_sec is absorbed into the longer of its two neighbours
        (its time interval is simply annexed — the absorbing neighbour's
        boundary is extended).  After each absorb, same-language runs
        are merged to prevent language repetitions.

        Pass B — Oscillation suppression
        -----------------------------------
        Detects A -> B -> A triplets where the middle B segment is shorter
        than 2 x min_seg_sec.  The triplet is collapsed into a single A.
        Runs until no more such triplets exist.  This handles the common
        Hinglish pattern where a speaker briefly switches language for a
        single word, then immediately returns.

        Returns
        -------
        Clean list of segment dicts (no "_n_frames" key):
            [{"start": float, "end": float, "lang": "hi"|"en"}, ...]
        """
        if not segments:
            return []

        segs = [s.copy() for s in segments]

        # ── Pass A: absorb short segments ─────────────────────────────────
        changed = True
        while changed:
            changed = False
            new_segs: List[Dict] = []
            i = 0

            while i < len(segs):
                seg = segs[i]
                dur = seg["end"] - seg["start"]

                if dur < self.min_seg_sec and len(segs) > 1:
                    prev     = new_segs[-1] if new_segs else None
                    nxt      = segs[i + 1] if i + 1 < len(segs) else None
                    prev_dur = (prev["end"] - prev["start"]) if prev else -1
                    nxt_dur  = (nxt["end"]  - nxt["start"])  if nxt  else -1

                    if prev is not None and (nxt is None or prev_dur >= nxt_dur):
                        # Absorb into previous: extend end boundary
                        new_segs[-1]["end"] = seg["end"]
                    elif nxt is not None:
                        # Absorb into next: pull start boundary back
                        segs[i + 1]["start"] = seg["start"]
                    else:
                        new_segs.append(seg)  # only segment left, keep it

                    changed = True
                else:
                    new_segs.append(seg)

                i += 1

            # Collapse any same-language neighbours created by absorptions
            segs = self._merge_same_lang(new_segs)

        # ── Pass B: suppress A→B→A oscillations ──────────────────────────
        segs = self._suppress_oscillations(segs)

        # ── Strip internal metadata key, round timestamps ─────────────────
        output: List[Dict] = []
        for seg in segs:
            output.append({
                "start": round(float(seg["start"]), 3),
                "end":   round(float(seg["end"]),   3),
                "lang":  seg["lang"],
            })

        log.info("Refinement complete | %d final segments", len(output))
        return output

    def _merge_same_lang(self, segments: List[Dict]) -> List[Dict]:
        """Collapse consecutive segments that share the same language label."""
        if not segments:
            return []
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            if seg["lang"] == merged[-1]["lang"]:
                merged[-1]["end"] = seg["end"]
                merged[-1]["_n_frames"] = (
                    merged[-1].get("_n_frames", 1) + seg.get("_n_frames", 1)
                )
            else:
                merged.append(seg.copy())
        return merged

    def _suppress_oscillations(self, segments: List[Dict]) -> List[Dict]:
        """
        Eliminate rapid A→B→A patterns.

        A triplet qualifies for suppression when:
          - segments[i].lang == segments[i+2].lang  (same language on both sides)
          - segments[i+1] duration < 2 × min_seg_sec  (the middle is short)

        The three segments are merged into one segment labelled A.
        Repeats until no qualifying triplets remain.
        """
        changed = True
        segs = [s.copy() for s in segments]

        while changed:
            changed = False
            new_segs: List[Dict] = []
            i = 0

            while i < len(segs):
                if (
                    i + 2 < len(segs)
                    and segs[i]["lang"] == segs[i + 2]["lang"]
                    and (segs[i + 1]["end"] - segs[i + 1]["start"])
                        < self.min_seg_sec * 2
                ):
                    # Collapse triplet into a single A segment
                    new_segs.append({
                        "start":     segs[i]["start"],
                        "end":       segs[i + 2]["end"],
                        "lang":      segs[i]["lang"],
                        "_n_frames": (
                            segs[i].get("_n_frames", 1)
                            + segs[i + 1].get("_n_frames", 1)
                            + segs[i + 2].get("_n_frames", 1)
                        ),
                    })
                    i += 3
                    changed = True
                else:
                    new_segs.append(segs[i])
                    i += 1

            segs = self._merge_same_lang(new_segs)

        return segs

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Main public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        audio:     "str | np.ndarray | torch.Tensor",
        source_sr: Optional[int] = None,
    ) -> List[Dict]:
        """
        Full pipeline: audio → language segments.

        Parameters
        ----------
        audio     : WAV file path, NumPy array, or Torch waveform tensor
        source_sr : sample rate of the input (needed for array/tensor inputs)

        Returns
        -------
        List[Dict]
            [{"start": 0.00, "end": 2.30, "lang": "en"},
             {"start": 2.30, "end": 5.10, "lang": "hi"}, ...]
        """
        self.load_model()

        # Step 1: load, resample, normalise
        waveform, _ = self.preprocess_audio(audio, source_sr)

        # Step 2: sliding-window framing
        windows, timestamps = self.frame_audio(waveform)
        if not windows:
            log.warning("Audio too short to produce any frames — returning empty.")
            return []

        # Step 3: per-frame SpeechBrain inference
        predictions = self.predict_frames(windows, timestamps)

        # Step 4: Gaussian + median smoothing
        predictions = self.smooth_predictions(predictions)

        # Step 5: run-length merge → raw segments
        segments = self.merge_segments(predictions)

        # Step 6: absorb short segments + suppress oscillations
        segments = self.refine_timestamps(segments)

        return segments

    # ─────────────────────────────────────────────────────────────────────────
    # Bonus: timeline visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def plot_timeline(
        self,
        segments:       List[Dict],
        total_duration: Optional[float] = None,
        predictions:    Optional[List[FramePrediction]] = None,
        title:          str = "Language ID Timeline — Hinglish Audio",
        save_path:      Optional[str] = None,
    ) -> None:
        """
        Two-panel visualisation:
          Top panel    — smoothed per-frame probability curves P(hi) and P(en)
                         (only rendered when 'predictions' is provided)
          Bottom panel — colour-coded language segment timeline

        Parameters
        ----------
        segments       : output of predict()
        total_duration : audio length in seconds; auto-inferred if None
        predictions    : FramePrediction list from predict_frames() for top panel
        title          : figure title
        save_path      : file path to save the figure (PNG/PDF); shows if None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.ticker import MultipleLocator
        except ImportError:
            log.error("matplotlib is required for plot_timeline(). "
                      "Install with: pip install matplotlib")
            return

        if not segments:
            log.warning("No segments to plot.")
            return

        total_dur = total_duration or segments[-1]["end"]
        COLORS = {"hi": "#E76F51", "en": "#2A9D8F"}
        LABELS = {"hi": "Hindi",   "en": "English"}

        has_probs = predictions is not None and len(predictions) > 0
        n_panels  = 2 if has_probs else 1

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(15, 4 * n_panels),
            gridspec_kw={"height_ratios": [2, 1] if has_probs else [1]},
            sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        # ── Top panel: probability curves ────────────────────────────────
        if has_probs:
            ax  = axes[0]
            mid = np.array([(p.start_sec + p.end_sec) / 2 for p in predictions])
            hi  = np.array([p.smooth_prob_hi for p in predictions])
            en  = np.array([p.smooth_prob_en for p in predictions])

            ax.fill_between(mid, hi, alpha=0.25, color=COLORS["hi"])
            ax.fill_between(mid, en, alpha=0.25, color=COLORS["en"])
            ax.plot(mid, hi, color=COLORS["hi"], lw=2.0, label="P(Hindi)")
            ax.plot(mid, en, color=COLORS["en"], lw=2.0, label="P(English)")
            ax.axhline(0.5, color="#555", lw=0.9, ls="--", alpha=0.6,
                       label="Decision boundary")

            # Mark switch boundaries as vertical dashed lines
            for seg in segments[1:]:
                ax.axvline(seg["start"], color="#888", lw=0.8, ls=":", alpha=0.7)

            ax.set_ylabel("Language Probability", fontsize=11)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlim(0, total_dur)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
            ax.grid(axis="x", alpha=0.25, ls=":")
            ax.spines[["top", "right"]].set_visible(False)

        # ── Bottom panel: coloured segment bar ───────────────────────────
        ax_seg = axes[-1]
        ax_seg.set_xlim(0, total_dur)
        ax_seg.set_ylim(0, 1)
        ax_seg.set_yticks([])

        for seg in segments:
            dur   = seg["end"] - seg["start"]
            color = COLORS.get(seg["lang"], "#999")

            # Draw rounded rectangle for each segment
            rect = mpatches.FancyBboxPatch(
                (seg["start"], 0.08), dur, 0.84,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor="white",
                linewidth=1.4, alpha=0.92,
            )
            ax_seg.add_patch(rect)

            # Label inside the block if wide enough to fit text
            if dur > 0.35:
                ax_seg.text(
                    seg["start"] + dur / 2, 0.5,
                    LABELS.get(seg["lang"], seg["lang"]),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )

            # Time-stamp at block left edge (only for wider blocks)
            if dur > 0.6:
                ax_seg.text(
                    seg["start"] + 0.02, 0.15,
                    f"{seg['start']:.1f}s",
                    ha="left", va="bottom",
                    fontsize=7, color="white", alpha=0.85,
                )

        # X-axis formatting
        tick_step = max(0.5, round(total_dur / 20, 1))
        ticks = np.arange(0, total_dur + tick_step, tick_step)
        ax_seg.set_xticks(ticks)
        ax_seg.set_xticklabels(
            [f"{t:.1f}s" for t in ticks], fontsize=8, rotation=30, ha="right"
        )
        ax_seg.set_xlabel("Time (seconds)", fontsize=11)
        ax_seg.set_ylabel("Language", fontsize=11)
        ax_seg.spines[["top", "right", "left"]].set_visible(False)

        if not has_probs:
            ax_seg.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Legend for segment colours
        legend_patches = [
            mpatches.Patch(facecolor=COLORS[k], label=LABELS[k], alpha=0.9)
            for k in COLORS
        ]
        ax_seg.legend(
            handles=legend_patches, loc="lower right",
            fontsize=9, framealpha=0.85,
        )

        plt.tight_layout(h_pad=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Timeline saved → %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # Repr
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not yet loaded"
        return (
            f"LIDModule("
            f"model={self.model_name!r}, "
            f"window={self.window_sec}s, hop={self.hop_sec}s, "
            f"smooth={self.smooth_window} frames, "
            f"min_seg={self.min_seg_sec}s, "
            f"device={self.device!r}, "
            f"status={status})"
        )

"""
stt/transcriber.py
===================
Segment-aware Speech-to-Text using pretrained OpenAI Whisper.

Design
------
  - Accepts raw audio + LID segment list  (output of LIDModule.predict())
  - Runs Whisper independently on each LID segment with the correct
    language hint → avoids cross-lingual contamination in code-switched audio
  - Returns a flat, timestamp-aligned transcript merging all segments
  - Optionally applies ConstrainedDecoder logit bias during generation

Pipeline
--------
  LID segments + audio file
        │
        ▼
  _slice_segment()       — extract waveform slice per LID segment
        │
        ▼
  _build_decode_options()— set language, task, timestamps per segment
        │
        ▼
  Whisper.transcribe()   — forced language decoding per segment
        │
        ▼
  _merge_results()       — reconcile Whisper word-timestamps across segs
        │
        ▼
  List[WordToken]        — word-level timestamps + language + confidence
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

TARGET_SR = 16_000   # Whisper's required sample rate

# Whisper language codes for our target languages
LANG_TO_WHISPER = {"hi": "hi", "en": "en", "sil": "en"}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordToken:
    """Single word with timing and provenance."""
    word:       str
    start:      float          # seconds from beginning of original audio
    end:        float
    lang:       str            # "hi" or "en"
    confidence: float          # avg log-prob from Whisper (0-1 range)
    segment_id: int            # which LID segment produced this word


@dataclass
class TranscriptSegment:
    """Full transcript for one LID-guided segment."""
    segment_id:   int
    start:        float
    end:          float
    lang:         str
    text:         str          # raw Whisper transcript
    words:        List[WordToken] = field(default_factory=list)
    avg_logprob:  float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    """Final output of Transcriber.transcribe()."""
    full_text:   str
    segments:    List[TranscriptSegment]
    words:       List[WordToken]           # flat, time-sorted word list
    duration:    float

    def to_dict(self) -> Dict:
        return {
            "full_text": self.full_text,
            "duration":  self.duration,
            "words": [
                {
                    "word":       w.word,
                    "start":      round(w.start, 3),
                    "end":        round(w.end,   3),
                    "lang":       w.lang,
                    "confidence": round(w.confidence, 4),
                }
                for w in self.words
            ],
            "segments": [
                {
                    "id":    s.segment_id,
                    "start": round(s.start, 3),
                    "end":   round(s.end,   3),
                    "lang":  s.lang,
                    "text":  s.text,
                }
                for s in self.segments
            ],
        }

    def __str__(self) -> str:
        lines = [f"[Full transcript]\n{self.full_text}\n",
                 f"[Duration] {self.duration:.2f}s",
                 "\n[Word-level timestamps]"]
        for w in self.words:
            lines.append(
                f"  {w.start:6.2f}s – {w.end:6.2f}s  [{w.lang}]  {w.word!r}"
                f"  (conf={w.confidence:.3f})"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Transcriber
# ─────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """
    Segment-aware ASR transcriber for code-switched Hinglish audio.

    Parameters
    ----------
    model_size      : Whisper model size: "tiny","base","small","medium","large-v3"
    device          : "cpu", "cuda", or None (auto-detect)
    constrained_decoder : optional ConstrainedDecoder for logit biasing
    word_timestamps : enable Whisper word-level timing (requires whisper>=20230918)
    no_speech_threshold : suppress segments with high silence probability
    temperature     : decoding temperature (0 = greedy)
    """

    def __init__(
        self,
        model_size:           str = "base",
        device:               Optional[str] = None,
        constrained_decoder=  None,         # ConstrainedDecoder | None
        word_timestamps:      bool = True,
        no_speech_threshold:  float = 0.6,
        temperature:          float = 0.0,
    ):
        self.model_size          = model_size
        self.device              = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.constrained_decoder = constrained_decoder
        self.word_timestamps     = word_timestamps
        self.no_speech_threshold = no_speech_threshold
        self.temperature         = temperature

        self._model   = None
        self._options = None

        log.info(
            "Transcriber init | model=whisper-%s | device=%s | word_ts=%s",
            model_size, self.device, word_timestamps,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Model loading (lazy)
    # ─────────────────────────────────────────────────────────────────────

    def load_model(self) -> "Transcriber":
        if self._model is not None:
            return self
        try:
            import whisper
        except ImportError:
            raise ImportError("Run: pip install openai-whisper")

        log.info("Loading Whisper '%s' ...", self.model_size)
        self._model = whisper.load_model(self.model_size, device=self.device)
        log.info("Whisper loaded.")
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Audio loading + slicing
    # ─────────────────────────────────────────────────────────────────────

    def _load_waveform(self, source: "str | np.ndarray | torch.Tensor") -> np.ndarray:
        """
        Load audio → mono float32 numpy array at 16 kHz.
        Whisper's internal pad_or_trim expects numpy.
        """
        if isinstance(source, (str, Path)):
            wav, sr = torchaudio.load(str(source))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != TARGET_SR:
                wav = T.Resample(sr, TARGET_SR)(wav)
            return wav.squeeze(0).numpy()

        elif isinstance(source, np.ndarray):
            arr = source.astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr

        elif isinstance(source, torch.Tensor):
            t = source.float()
            if t.ndim > 1:
                t = t.mean(0)
            return t.numpy()

        raise TypeError(f"Unsupported audio type: {type(source)}")

    def _slice_segment(
        self,
        waveform: np.ndarray,
        start_sec: float,
        end_sec: float,
        pad_sec: float = 0.1,
    ) -> np.ndarray:
        """
        Extract a time-slice of the waveform with small context padding.
        The padding helps Whisper decode words at segment boundaries.
        """
        start_sample = max(0, int((start_sec - pad_sec) * TARGET_SR))
        end_sample   = min(len(waveform), int((end_sec + pad_sec) * TARGET_SR))
        return waveform[start_sample:end_sample]

    # ─────────────────────────────────────────────────────────────────────
    # Per-segment decoding
    # ─────────────────────────────────────────────────────────────────────

    def _decode_segment(
        self,
        clip:       np.ndarray,
        lang:       str,
        seg_offset: float,
        seg_id:     int,
    ) -> Optional[TranscriptSegment]:
        """
        Run Whisper on a single audio clip with forced language.

        Parameters
        ----------
        clip       : mono float32 numpy array (segment waveform)
        lang       : language hint from LID ("hi" or "en")
        seg_offset : start time of this clip in the original audio (seconds)
        seg_id     : LID segment index

        Returns None if the segment is classified as silence / no-speech.
        """
        import whisper

        whisper_lang = LANG_TO_WHISPER.get(lang, "hi")

        # Build decoding options
        # logit_filters can host our ConstrainedDecoder if provided
        decode_options = whisper.DecodingOptions(
            language=whisper_lang,
            task="transcribe",
            without_timestamps=not self.word_timestamps,
            temperature=self.temperature,
        )

        # Pad/trim to Whisper's 30 s mel window
        mel = whisper.log_mel_spectrogram(clip).to(self.device)
        mel = whisper.pad_or_trim(mel, whisper.audio.N_FRAMES)

        # Whisper decode
        result = whisper.decode(self._model, mel, decode_options)

        # Silence gate
        if result.no_speech_prob > self.no_speech_threshold:
            log.debug("Segment %d suppressed (no_speech_prob=%.3f)", seg_id, result.no_speech_prob)
            return None

        # Build word tokens
        words: List[WordToken] = []

        if self.word_timestamps and hasattr(result, "words") and result.words:
            for w in result.words:
                # w.start / w.end are relative to the clip; add seg_offset
                words.append(WordToken(
                    word       = w.word.strip(),
                    start      = w.start + seg_offset,
                    end        = w.end   + seg_offset,
                    lang       = lang,
                    confidence = float(np.exp(w.probability)) if hasattr(w, "probability")
                                 else float(np.exp(result.avg_logprob)),
                    segment_id = seg_id,
                ))
        else:
            # No word-level timestamps: assign all text to the segment span
            duration = len(clip) / TARGET_SR
            words.append(WordToken(
                word       = result.text.strip(),
                start      = seg_offset,
                end        = seg_offset + duration,
                lang       = lang,
                confidence = float(np.exp(result.avg_logprob)),
                segment_id = seg_id,
            ))

        return TranscriptSegment(
            segment_id    = seg_id,
            start         = seg_offset,
            end           = seg_offset + len(clip) / TARGET_SR,
            lang          = lang,
            text          = result.text.strip(),
            words         = words,
            avg_logprob   = float(result.avg_logprob),
            no_speech_prob= float(result.no_speech_prob),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Constrained decoding hook
    # ─────────────────────────────────────────────────────────────────────

    def _apply_constrained_decoding(
        self,
        clip: np.ndarray,
        lang: str,
        seg_offset: float,
        seg_id: int,
    ) -> Optional[TranscriptSegment]:
        """
        Use ConstrainedDecoder to bias Whisper's token logits before greedy
        selection.  Falls back to standard _decode_segment if decoder absent.
        """
        if self.constrained_decoder is None:
            return self._decode_segment(clip, lang, seg_offset, seg_id)

        import whisper

        whisper_lang = LANG_TO_WHISPER.get(lang, "hi")
        mel = whisper.log_mel_spectrogram(clip).to(self.device)

        # FIX: record the true clip duration BEFORE pad_or_trim overwrites it.
        # mel.shape[-1] after pad_or_trim is always N_FRAMES (3000 = 30 s * 100 fps)
        # regardless of how short the segment actually is.  We must capture the
        # real duration here so generate() can build accurate word timestamps.
        actual_clip_duration = clip.shape[0] / TARGET_SR   # seconds

        mel = whisper.pad_or_trim(mel, whisper.audio.N_FRAMES)

        # Use the constrained decoder's generate method
        text, words, avg_lp = self.constrained_decoder.generate(
            model           = self._model,
            mel             = mel,
            language        = whisper_lang,
            word_timestamps = self.word_timestamps,
            seg_offset      = seg_offset,
            seg_id          = seg_id,
            lang            = lang,
            seg_duration    = actual_clip_duration,   # FIX: pass real duration
        )

        if text is None:
            return None

        return TranscriptSegment(
            segment_id  = seg_id,
            start       = seg_offset,
            end         = seg_offset + actual_clip_duration,
            lang        = lang,
            text        = text,
            words       = words,
            avg_logprob = avg_lp,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Main transcribe()
    # ─────────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio:        "str | np.ndarray | torch.Tensor",
        lid_segments: Optional[List[Dict]] = None,
        default_lang: str = "hi",
    ) -> TranscriptionResult:
        """
        Transcribe audio using LID-guided language hints.

        Parameters
        ----------
        audio        : file path, numpy array, or torch tensor (16 kHz mono)
        lid_segments : output of LIDModule.predict()
                       [{"start": float, "end": float, "lang": "hi"|"en"}, ...]
                       If None, treats whole audio as default_lang.
        default_lang : fallback language if lid_segments is None

        Returns
        -------
        TranscriptionResult with full_text, per-segment and word-level output
        """
        self.load_model()

        # Load full waveform
        waveform = self._load_waveform(audio)
        duration = len(waveform) / TARGET_SR
        log.info("Transcribing %.2f s of audio", duration)

        # If no LID segments provided, treat whole audio as one segment
        if not lid_segments:
            lid_segments = [{"start": 0.0, "end": duration, "lang": default_lang}]

        transcript_segments: List[TranscriptSegment] = []
        all_words:           List[WordToken]          = []

        for seg_id, lid_seg in enumerate(lid_segments):
            start = lid_seg["start"]
            end   = lid_seg["end"]
            lang  = lid_seg["lang"]

            log.info(
                "  Segment %d/%d  [%.2fs – %.2fs]  lang=%s",
                seg_id + 1, len(lid_segments), start, end, lang,
            )

            # Skip extremely short segments (< 0.3 s)
            if end - start < 0.3:
                log.debug("  Skipping segment %d: too short (%.2fs)", seg_id, end-start)
                continue

            clip = self._slice_segment(waveform, start, end)

            # Decode — with or without constrained decoding
            ts = self._apply_constrained_decoding(clip, lang, start, seg_id)
            if ts is None:
                continue

            transcript_segments.append(ts)
            all_words.extend(ts.words)

        # Sort words by start time (segments may have overlap from padding)
        all_words.sort(key=lambda w: w.start)

        # Deduplicate words at segment boundaries (caused by overlap padding)
        all_words = self._deduplicate_words(all_words)

        # Build full text preserving language ordering
        full_text = self._build_full_text(transcript_segments)

        result = TranscriptionResult(
            full_text = full_text,
            segments  = transcript_segments,
            words     = all_words,
            duration  = duration,
        )

        log.info("Transcription complete | %d words | %d segments",
                 len(all_words), len(transcript_segments))
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Post-processing helpers
    # ─────────────────────────────────────────────────────────────────────

    def _deduplicate_words(self, words: List[WordToken]) -> List[WordToken]:
        """
        Remove duplicate words at segment boundaries introduced by the
        overlap padding in _slice_segment().
        Two words are considered duplicates if they have the same text AND
        their time intervals overlap by more than 50 %.
        """
        if len(words) <= 1:
            return words

        kept: List[WordToken] = [words[0]]
        for curr in words[1:]:
            prev = kept[-1]
            # Compute overlap ratio
            overlap_start = max(prev.start, curr.start)
            overlap_end   = min(prev.end,   curr.end)
            overlap       = max(0.0, overlap_end - overlap_start)
            curr_dur      = max(curr.end - curr.start, 1e-6)

            if curr.word.strip().lower() == prev.word.strip().lower() \
               and overlap / curr_dur > 0.5:
                continue    # duplicate — skip
            kept.append(curr)

        return kept

    def _build_full_text(self, segments: List[TranscriptSegment]) -> str:
        """
        Concatenate segment texts with a space separator.
        Adds a subtle language-change marker ‖ for readability.
        """
        if not segments:
            return ""
        parts = [segments[0].text]
        for i in range(1, len(segments)):
            sep = " ‖ " if segments[i].lang != segments[i-1].lang else " "
            parts.append(sep + segments[i].text)
        return "".join(parts).strip()

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return (
            f"Transcriber(model=whisper-{self.model_size}, "
            f"device={self.device!r}, "
            f"constrained={self.constrained_decoder is not None}, "
            f"status={loaded})"
        )

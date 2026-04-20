"""
tts/speaker_embedder.py
=======================
Speaker embedding extraction for voice cloning pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)

EmbeddingBackend = Literal["resemblyzer", "speechbrain"]
AudioInput = Union[str, Path, np.ndarray, torch.Tensor]


class SpeakerEmbedder:
    """
    Extract a fixed-dimensional speaker embedding from a reference voice sample.

    Supports:
    - `resemblyzer` d-vectors
    - `speechbrain` x-vectors / ECAPA speaker embeddings
    """

    def __init__(
        self,
        backend: EmbeddingBackend = "resemblyzer",
        target_sr: int = 16_000,
        max_duration_sec: float = 60.0,
        device: Optional[str] = None,
        speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        speechbrain_savedir: str = "pretrained_models/speaker_embedding",
    ) -> None:
        self.backend = backend
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speechbrain_model = speechbrain_model
        self.speechbrain_savedir = speechbrain_savedir

        self._resemblyzer_encoder = None
        self._speechbrain_encoder = None

    def extract_embedding(
        self,
        source: AudioInput,
        source_sr: Optional[int] = None,
    ) -> np.ndarray:
        """Load a voice sample, keep up to 60 seconds, and return an embedding vector."""
        waveform = self._load_audio(source, source_sr)

        if self.backend == "resemblyzer":
            return self._extract_with_resemblyzer(waveform)
        if self.backend == "speechbrain":
            return self._extract_with_speechbrain(waveform)

        raise ValueError(f"Unsupported speaker embedding backend: {self.backend}")

    def _load_audio(self, source: AudioInput, source_sr: Optional[int]) -> torch.Tensor:
        """Load audio as mono float32 tensor of shape (1, T) at target sample rate."""
        if isinstance(source, (str, Path)):
            waveform, sr = torchaudio.load(str(source))
        elif isinstance(source, np.ndarray):
            array = source.astype(np.float32)
            waveform = torch.from_numpy(array)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = source_sr or self.target_sr
        elif isinstance(source, torch.Tensor):
            waveform = source.detach().float().cpu()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            sr = source_sr or self.target_sr
        else:
            raise TypeError(f"Unsupported audio source type: {type(source)}")

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        max_samples = int(self.max_duration_sec * self.target_sr)
        if waveform.shape[1] > max_samples:
            log.info(
                "Reference audio longer than %.1f s; trimming to first %.1f s.",
                self.max_duration_sec,
                self.max_duration_sec,
            )
            waveform = waveform[:, :max_samples]

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform.contiguous().float()

    def _extract_with_resemblyzer(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract a d-vector using Resemblyzer."""
        if self._resemblyzer_encoder is None:
            try:
                from resemblyzer import VoiceEncoder
            except ImportError as exc:
                raise ImportError(
                    "Resemblyzer is not installed. Run: pip install resemblyzer"
                ) from exc
            self._resemblyzer_encoder = VoiceEncoder(device=self.device)

        wav_np = waveform.squeeze(0).cpu().numpy()
        embedding = self._resemblyzer_encoder.embed_utterance(wav_np)
        return np.asarray(embedding, dtype=np.float32)

    def _extract_with_speechbrain(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract an x-vector-style speaker embedding using SpeechBrain."""
        if self._speechbrain_encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError as exc:
                raise ImportError(
                    "SpeechBrain is not installed. Run: pip install speechbrain>=1.0"
                ) from exc

            self._speechbrain_encoder = EncoderClassifier.from_hparams(
                source=self.speechbrain_model,
                savedir=self.speechbrain_savedir,
                run_opts={"device": self.device},
            )

        waveform = waveform.to(self.device)
        embedding = self._speechbrain_encoder.encode_batch(waveform)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def __call__(self, source: AudioInput, source_sr: Optional[int] = None) -> np.ndarray:
        return self.extract_embedding(source, source_sr=source_sr)

    def __repr__(self) -> str:
        return (
            f"SpeakerEmbedder(backend={self.backend!r}, target_sr={self.target_sr}, "
            f"max_duration_sec={self.max_duration_sec}, device={self.device!r})"
        )

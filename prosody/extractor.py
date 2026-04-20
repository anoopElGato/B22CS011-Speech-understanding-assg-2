"""
prosody/extractor.py
====================
Pitch and energy contour extraction with aligned timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]


@dataclass
class ProsodyContour:
    """Aligned prosodic features for a waveform."""

    timestamps: np.ndarray
    f0_curve: np.ndarray
    energy_curve: np.ndarray
    sample_rate: int
    hop_length: int
    frame_length: int

    def to_dict(self) -> dict:
        return {
            "timestamps": self.timestamps.tolist(),
            "f0_curve": self.f0_curve.tolist(),
            "energy_curve": self.energy_curve.tolist(),
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
        }


class ProsodyExtractor:
    """
    Extract F0 and energy contours aligned to frame timestamps.

    F0 backends:
    - `pyworld` when available
    - `librosa.yin` fallback otherwise
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        frame_length: int = 1024,
        hop_length: int = 256,
        f0_backend: str = "pyworld",
        f0_floor: float = 50.0,
        f0_ceil: float = 500.0,
    ) -> None:
        self.target_sr = target_sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.f0_backend = f0_backend
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil

    def extract(
        self,
        source: AudioInput,
        source_sr: Optional[int] = None,
    ) -> ProsodyContour:
        """Extract F0 curve, energy curve, and timestamps from audio."""
        waveform = self._load_audio(source, source_sr)
        f0_curve, timestamps = self._extract_f0(waveform)
        energy_curve = self._extract_energy(waveform, num_frames=len(f0_curve))
        timestamps = self._align_length(timestamps, len(f0_curve))
        energy_curve = self._align_length(energy_curve, len(f0_curve))

        return ProsodyContour(
            timestamps=timestamps.astype(np.float32),
            f0_curve=f0_curve.astype(np.float32),
            energy_curve=energy_curve.astype(np.float32),
            sample_rate=self.target_sr,
            hop_length=self.hop_length,
            frame_length=self.frame_length,
        )

    def _load_audio(self, source: AudioInput, source_sr: Optional[int]) -> np.ndarray:
        """Load audio as mono float32 NumPy array at target sample rate."""
        if isinstance(source, (str, Path)):
            waveform, sr = torchaudio.load(str(source))
        elif isinstance(source, np.ndarray):
            waveform = torch.from_numpy(source.astype(np.float32))
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

        wav = waveform.squeeze(0).numpy()
        if wav.size == 0:
            raise ValueError("Audio is empty.")
        return wav.astype(np.float64)

    def _extract_f0(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 using pyworld if available, otherwise librosa.yin."""
        if self.f0_backend == "pyworld":
            try:
                import pyworld as pw

                frame_period_ms = (self.hop_length / self.target_sr) * 1000.0
                f0, t = pw.dio(
                    waveform,
                    fs=self.target_sr,
                    f0_floor=self.f0_floor,
                    f0_ceil=self.f0_ceil,
                    frame_period=frame_period_ms,
                )
                f0 = pw.stonemask(waveform, f0, t, self.target_sr)
                return f0.astype(np.float32), t.astype(np.float32)
            except Exception:
                log.warning("pyworld unavailable or failed; falling back to librosa.yin.", exc_info=True)

        f0 = librosa.yin(
            waveform.astype(np.float32),
            fmin=self.f0_floor,
            fmax=self.f0_ceil,
            sr=self.target_sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        timestamps = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.target_sr,
            hop_length=self.hop_length,
        )
        return f0.astype(np.float32), timestamps.astype(np.float32)

    def _extract_energy(self, waveform: np.ndarray, num_frames: int) -> np.ndarray:
        """Extract RMS energy contour on the same analysis grid."""
        rms = librosa.feature.rms(
            y=waveform.astype(np.float32),
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            center=True,
        ).squeeze(0)
        return self._align_length(rms, num_frames).astype(np.float32)

    @staticmethod
    def _align_length(values: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or trim a 1D array to the target frame count."""
        if len(values) == target_len:
            return values
        if len(values) > target_len:
            return values[:target_len]
        if len(values) == 0:
            return np.zeros(target_len, dtype=np.float32)
        pad_width = target_len - len(values)
        return np.pad(values, (0, pad_width), mode="edge")

    def save_contours(self, contour: ProsodyContour, output_path: Union[str, Path]) -> Path:
        """Persist extracted contours as a NumPy archive."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            timestamps=contour.timestamps,
            f0_curve=contour.f0_curve,
            energy_curve=contour.energy_curve,
            sample_rate=contour.sample_rate,
            hop_length=contour.hop_length,
            frame_length=contour.frame_length,
        )
        return path

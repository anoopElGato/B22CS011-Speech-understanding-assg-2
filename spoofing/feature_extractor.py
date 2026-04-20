"""
spoofing/feature_extractor.py
=============================
Frame-level LFCC and CQCC feature extraction for anti-spoofing pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.fft import dct

log = logging.getLogger(__name__)

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]
FeatureType = Literal["lfcc", "cqcc"]


class AntiSpoofFeatureExtractor:
    """
    Extract frame-based LFCC or CQCC features for spoofing detection.

    The extractor:
    - loads audio from file / NumPy / Torch inputs
    - converts to mono and target sample rate
    - computes a frame-level cepstral matrix
    - optionally normalizes each coefficient dimension

    Output shape is always `(num_frames, num_coeffs)`.
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        frame_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        n_lfcc: int = 20,
        n_lfcc_filters: int = 40,
        n_cqcc: int = 20,
        cqt_bins: int = 84,
        bins_per_octave: int = 12,
        fmin: float = 32.7,
        normalize: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.target_sr = target_sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_lfcc = n_lfcc
        self.n_lfcc_filters = n_lfcc_filters
        self.n_cqcc = n_cqcc
        self.cqt_bins = cqt_bins
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._lfcc_transform: Optional[T.LFCC] = None

    def extract(
        self,
        source: AudioInput,
        feature_type: FeatureType = "lfcc",
        source_sr: Optional[int] = None,
    ) -> np.ndarray:
        """Extract a normalized frame-level feature matrix."""
        waveform = self._load_audio(source, source_sr)

        if feature_type == "lfcc":
            features = self.extract_lfcc(waveform)
        elif feature_type == "cqcc":
            features = self.extract_cqcc(waveform)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")

        if self.normalize:
            features = self.normalize_features(features)

        return features.astype(np.float32)

    def extract_lfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Extract frame-based LFCC features."""
        tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        transform = self._get_lfcc_transform().to(self.device)

        with torch.no_grad():
            coeffs = transform(tensor)

        matrix = coeffs.squeeze(0).transpose(0, 1).detach().cpu().numpy()
        return self._ensure_2d(matrix, self.n_lfcc)

    def extract_cqcc(self, waveform: np.ndarray) -> np.ndarray:
        """Extract frame-based CQCC features on a log-frequency axis."""
        tensor = torch.from_numpy(waveform.astype(np.float32))
        window = torch.hann_window(self.frame_length)
        stft = torch.stft(
            input=tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = (stft.abs() ** 2).cpu().numpy()
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.target_sr)
        target_freqs = self.fmin * (2.0 ** (np.arange(self.cqt_bins) / self.bins_per_octave))
        target_freqs = np.clip(target_freqs, self.fmin, self.target_sr / 2.0)

        log_power = np.log(np.maximum(power, 1e-10))
        constant_q_log_power = np.stack(
            [
                np.interp(target_freqs, freqs, log_power[:, frame_idx])
                for frame_idx in range(log_power.shape[1])
            ],
            axis=1,
        )
        cepstra = dct(constant_q_log_power, type=2, axis=0, norm="ortho")
        matrix = cepstra[: self.n_cqcc].T
        return self._ensure_2d(matrix, self.n_cqcc)

    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """Apply per-coefficient mean-variance normalization."""
        if features.size == 0:
            return features.astype(np.float32)

        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        normalized = (features - mean) / std
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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

        wav = waveform.squeeze(0).numpy().astype(np.float32)
        if wav.size == 0:
            raise ValueError("Audio is empty.")

        peak = float(np.max(np.abs(wav)))
        if peak > 0.0:
            wav = wav / peak

        return wav

    def _get_lfcc_transform(self) -> T.LFCC:
        if self._lfcc_transform is None:
            self._lfcc_transform = T.LFCC(
                sample_rate=self.target_sr,
                n_lfcc=self.n_lfcc,
                speckwargs={
                    "n_fft": self.n_fft,
                    "win_length": self.frame_length,
                    "hop_length": self.hop_length,
                    "center": True,
                    "power": 2.0,
                },
                n_filter=self.n_lfcc_filters,
                log_lf=True,
            )
        return self._lfcc_transform

    @staticmethod
    def _ensure_2d(matrix: np.ndarray, num_coeffs: int) -> np.ndarray:
        if matrix.ndim != 2:
            matrix = np.asarray(matrix).reshape(-1, num_coeffs)
        if matrix.shape[0] == 0:
            return np.zeros((1, num_coeffs), dtype=np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __call__(
        self,
        source: AudioInput,
        feature_type: FeatureType = "lfcc",
        source_sr: Optional[int] = None,
    ) -> np.ndarray:
        return self.extract(source, feature_type=feature_type, source_sr=source_sr)

    def __repr__(self) -> str:
        return (
            f"AntiSpoofFeatureExtractor(target_sr={self.target_sr}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length}, "
            f"n_lfcc={self.n_lfcc}, n_cqcc={self.n_cqcc}, normalize={self.normalize})"
        )

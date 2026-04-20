"""
preprocessing/audio_preprocessor.py
=====================================
Stage 1 of the Speech Processing Pipeline.

Responsibilities
----------------
  - Load a WAV (or any librosa-readable) file
  - Denoise  via spectral subtraction (built-in, zero extra deps)
            OR via DeepFilterNet (optional, GPU-friendly)
  - Normalise amplitude to a target dBFS level
  - Resample to 16 kHz
  - Save the clean waveform

Backends
--------
  DENOISE_BACKEND env-var / constructor arg:
    "spectral"  — classic spectral subtraction  (default, always available)
    "deepfilter" — DeepFilterNet inference       (requires `pip install deepfilternet`)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import librosa
import librosa.effects
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
DenoiseBackend = Literal["spectral", "deepfilter"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(waveform: np.ndarray, sr: int) -> Tuple[torch.Tensor, int]:
    """Convert a NumPy array (samples,) or (channels, samples) to torch tensor."""
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]          # (1, T)
    tensor = torch.from_numpy(waveform.astype(np.float32))
    return tensor, sr


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Squeeze channel dim and return (T,) NumPy array."""
    return tensor.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Spectral Subtraction
# ---------------------------------------------------------------------------

def spectral_subtraction(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_frames: int = 10,
    alpha: float = 2.0,
    beta: float = 0.002,
) -> np.ndarray:
    """
    Classic power-spectrum subtraction.

    Parameters
    ----------
    waveform    : mono float32 array (T,)
    sr          : sample rate
    n_fft       : FFT size
    hop_length  : hop in samples
    noise_frames: number of leading frames used to estimate noise PSD
    alpha       : over-subtraction factor  (higher → more aggressive)
    beta        : spectral floor (prevents musical noise going negative)

    Returns
    -------
    denoised waveform (T,) float32
    """
    # STFT
    D = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)
    power = magnitude ** 2

    # Noise PSD estimate from first N frames
    noise_psd = np.mean(power[:, :noise_frames], axis=1, keepdims=True)

    # Spectral subtraction with flooring
    clean_power = np.maximum(power - alpha * noise_psd, beta * power)
    clean_magnitude = np.sqrt(clean_power)

    # Reconstruct with original phase
    D_clean = clean_magnitude * np.exp(1j * phase)
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(waveform))
    return y_clean.astype(np.float32)


# ---------------------------------------------------------------------------
# DeepFilterNet wrapper
# ---------------------------------------------------------------------------

def deepfilter_denoise(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Run DeepFilterNet inference.
    Requires: pip install deepfilternet
    """
    try:
        from df.enhance import enhance, init_df  # type: ignore
    except ImportError as e:
        raise ImportError(
            "DeepFilterNet is not installed. "
            "Run:  pip install deepfilternet\n"
            "Or use backend='spectral' for zero-dep denoising."
        ) from e

    model, df_state, _ = init_df()
    tensor, _ = _to_tensor(waveform, sr)

    # DeepFilterNet expects (channels, samples) at its native 48 kHz
    if sr != df_state.sr():
        resampler = T.Resample(orig_freq=sr, new_freq=df_state.sr())
        tensor = resampler(tensor)

    enhanced = enhance(model, df_state, tensor)

    # Resample back to original sr if needed
    if sr != df_state.sr():
        resampler_back = T.Resample(orig_freq=df_state.sr(), new_freq=sr)
        enhanced = resampler_back(enhanced)

    return _to_numpy(enhanced)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """
    Modular audio preprocessing stage.

    Usage
    -----
    >>> ap = AudioPreprocessor(target_sr=16_000, backend="spectral")
    >>> ap.load_audio("noisy_speech.wav")
    >>> ap.denoise()
    >>> ap.normalize(target_dbfs=-23.0)
    >>> ap.save_audio("clean_speech.wav")
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        backend: DenoiseBackend = "spectral",
        mono: bool = True,
        # spectral-subtraction knobs
        ss_n_fft: int = 512,
        ss_hop_length: int = 128,
        ss_noise_frames: int = 10,
        ss_alpha: float = 2.0,
        ss_beta: float = 0.002,
    ):
        self.target_sr = target_sr
        self.backend: DenoiseBackend = (
            os.environ.get("DENOISE_BACKEND", backend).lower()  # type: ignore[assignment]
        )
        self.mono = mono

        # spectral-subtraction hyper-params
        self._ss_kwargs = dict(
            n_fft=ss_n_fft,
            hop_length=ss_hop_length,
            noise_frames=ss_noise_frames,
            alpha=ss_alpha,
            beta=ss_beta,
        )

        # internal state
        self._waveform: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._source_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def waveform(self) -> np.ndarray:
        if self._waveform is None:
            raise RuntimeError("No audio loaded. Call load_audio() first.")
        return self._waveform

    @property
    def sr(self) -> int:
        if self._sr is None:
            raise RuntimeError("No audio loaded. Call load_audio() first.")
        return self._sr

    # ------------------------------------------------------------------
    # Stage 1 — Load
    # ------------------------------------------------------------------

    def load_audio(self, path: str) -> "AudioPreprocessor":
        """
        Load an audio file.  Resamples to target_sr immediately.

        Parameters
        ----------
        path : str  Path to audio file (WAV, FLAC, MP3, …)

        Returns
        -------
        self  (for method chaining)
        """
        self._source_path = Path(path)
        log.info("Loading audio from %s", self._source_path)

        # librosa: returns float32, normalised to [-1, 1]
        waveform, sr = librosa.load(
            str(self._source_path),
            sr=self.target_sr,       # resamples on load
            mono=self.mono,
        )

        self._waveform = waveform.astype(np.float32)
        self._sr = self.target_sr

        duration = len(self._waveform) / self._sr
        log.info(
            "Loaded: sr=%d Hz | duration=%.2f s | shape=%s",
            self._sr, duration, self._waveform.shape,
        )
        return self

    # ------------------------------------------------------------------
    # Stage 2 — Denoise
    # ------------------------------------------------------------------

    def denoise(self) -> "AudioPreprocessor":
        """
        Apply the configured denoising backend in-place.

        Returns
        -------
        self
        """
        log.info("Denoising with backend='%s'", self.backend)

        if self.backend == "spectral":
            self._waveform = spectral_subtraction(
                self.waveform, self.sr, **self._ss_kwargs
            )

        elif self.backend == "deepfilter":
            self._waveform = deepfilter_denoise(self.waveform, self.sr)

        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                "Choose 'spectral' or 'deepfilter'."
            )

        log.info("Denoising complete.")
        return self

    # ------------------------------------------------------------------
    # Stage 3 — Normalise
    # ------------------------------------------------------------------

    def normalize(
        self,
        target_dbfs: float = -23.0,
        method: Literal["rms", "peak"] = "rms",
    ) -> "AudioPreprocessor":
        """
        Normalise amplitude to a target dBFS level.

        Parameters
        ----------
        target_dbfs : float  Target loudness in dBFS  (default: -23 LUFS-ish)
        method      : 'rms'  uses RMS-based gain;  'peak' uses peak normalisation

        Returns
        -------
        self
        """
        wav = self.waveform.copy()

        if method == "rms":
            rms = np.sqrt(np.mean(wav ** 2)) + 1e-9
            current_dbfs = 20.0 * np.log10(rms)
            gain_db = target_dbfs - current_dbfs
            gain_linear = 10.0 ** (gain_db / 20.0)
            log.info(
                "RMS normalise: current=%.1f dBFS → target=%.1f dBFS (gain=%.2f dB)",
                current_dbfs, target_dbfs, gain_db,
            )

        elif method == "peak":
            peak = np.max(np.abs(wav)) + 1e-9
            current_dbfs = 20.0 * np.log10(peak)
            gain_db = target_dbfs - current_dbfs
            gain_linear = 10.0 ** (gain_db / 20.0)
            log.info(
                "Peak normalise: current=%.1f dBFS → target=%.1f dBFS (gain=%.2f dB)",
                current_dbfs, target_dbfs, gain_db,
            )

        else:
            raise ValueError(f"Unknown normalisation method '{method}'.")

        wav = wav * gain_linear
        # Hard clip to [-1, 1] as safety net
        self._waveform = np.clip(wav, -1.0, 1.0).astype(np.float32)
        return self

    # ------------------------------------------------------------------
    # Stage 4 — Save
    # ------------------------------------------------------------------

    def save_audio(self, output_path: str, subtype: str = "PCM_16") -> Path:
        """
        Write the processed waveform to disk as a WAV file.

        Parameters
        ----------
        output_path : str   Destination file path (will be created)
        subtype     : str   soundfile subtype (default PCM_16 = 16-bit WAV)

        Returns
        -------
        Path  of the saved file
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out), self.waveform, self.sr, subtype=subtype)
        size_kb = out.stat().st_size / 1024
        log.info("Saved clean audio → %s  (%.1f KB)", out, size_kb)
        return out

    # ------------------------------------------------------------------
    # Convenience: one-shot class method
    # ------------------------------------------------------------------

    @classmethod
    def process(
        cls,
        input_path: str,
        output_path: str,
        target_sr: int = 16_000,
        backend: DenoiseBackend = "spectral",
        target_dbfs: float = -23.0,
    ) -> Path:
        """
        One-liner helper that runs the full preprocessing chain.

        >>> AudioPreprocessor.process("noisy.wav", "clean.wav")
        """
        return (
            cls(target_sr=target_sr, backend=backend)
            .load_audio(input_path)
            .denoise()
            .normalize(target_dbfs=target_dbfs)
            .save_audio(output_path)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snr_estimate(self) -> float:
        """
        Very rough SNR estimate using the ratio of signal power
        to the minimum-power frame (proxy for noise floor).
        """
        wav = self.waveform
        frame_size = int(0.02 * self.sr)   # 20 ms frames
        frames = [
            wav[i : i + frame_size]
            for i in range(0, len(wav) - frame_size, frame_size)
        ]
        powers = [np.mean(f ** 2) for f in frames if len(f) == frame_size]
        if not powers:
            return 0.0
        signal_power = np.percentile(powers, 95)
        noise_power  = np.percentile(powers, 5) + 1e-12
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def __repr__(self) -> str:
        src = self._source_path.name if self._source_path else "—"
        dur = f"{len(self._waveform)/self._sr:.2f}s" if self._waveform is not None else "—"
        return (
            f"AudioPreprocessor(src={src!r}, sr={self.target_sr}, "
            f"backend={self.backend!r}, duration={dur})"
        )

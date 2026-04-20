"""
prosody/mapper.py
=================
Dynamic Time Warping based prosody alignment and warping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from .extractor import AudioInput, ProsodyContour, ProsodyExtractor

log = logging.getLogger(__name__)


@dataclass
class ProsodyMappingResult:
    """Container for DTW-aligned prosody warping output."""

    source_contour: ProsodyContour
    synthesized_contour: ProsodyContour
    warped_f0_curve: np.ndarray
    warped_energy_curve: np.ndarray
    dtw_path: List[Tuple[int, int]]
    warped_waveform: np.ndarray
    sample_rate: int

    def to_dict(self) -> dict:
        return {
            "warped_f0_curve": self.warped_f0_curve.tolist(),
            "warped_energy_curve": self.warped_energy_curve.tolist(),
            "dtw_path": self.dtw_path,
            "sample_rate": self.sample_rate,
            "source_contour": self.source_contour.to_dict(),
            "synthesized_contour": self.synthesized_contour.to_dict(),
        }


class ProsodyMapper:
    """
    Align teacher prosody to synthesized speech with Dynamic Time Warping.

    Pipeline
    --------
    1. Extract F0 and energy contours from source and synthesized speech.
    2. Compute a DTW path on normalized prosodic features.
    3. Warp source F0 and energy onto the synthesized timeline.
    4. Apply the warped contours to the synthesized waveform.
    """

    def __init__(
        self,
        extractor: Optional[ProsodyExtractor] = None,
        target_sr: int = 16_000,
        frame_length: int = 1024,
        hop_length: int = 256,
        f0_backend: str = "pyworld",
    ) -> None:
        self.extractor = extractor or ProsodyExtractor(
            target_sr=target_sr,
            frame_length=frame_length,
            hop_length=hop_length,
            f0_backend=f0_backend,
        )
        self.target_sr = self.extractor.target_sr
        self.frame_length = self.extractor.frame_length
        self.hop_length = self.extractor.hop_length

    def map_prosody(
        self,
        source_audio: AudioInput,
        synthesized_audio: AudioInput,
        source_sr: Optional[int] = None,
        synthesized_sr: Optional[int] = None,
    ) -> ProsodyMappingResult:
        """Align and warp source prosody onto synthesized speech."""
        source_waveform = self.extractor._load_audio(source_audio, source_sr).astype(np.float32)
        synth_waveform = self.extractor._load_audio(synthesized_audio, synthesized_sr).astype(np.float32)

        source_contour = self.extractor.extract(source_waveform, source_sr=self.target_sr)
        synthesized_contour = self.extractor.extract(synth_waveform, source_sr=self.target_sr)

        dtw_path = self._compute_dtw_path(source_contour, synthesized_contour)
        warped_f0 = self._warp_curve(source_contour.f0_curve, len(synthesized_contour.f0_curve), dtw_path)
        warped_energy = self._warp_curve(
            source_contour.energy_curve,
            len(synthesized_contour.energy_curve),
            dtw_path,
        )

        warped_waveform = self._apply_warping(
            synth_waveform=synth_waveform,
            synthesized_contour=synthesized_contour,
            warped_f0=warped_f0,
            warped_energy=warped_energy,
        )

        return ProsodyMappingResult(
            source_contour=source_contour,
            synthesized_contour=synthesized_contour,
            warped_f0_curve=warped_f0.astype(np.float32),
            warped_energy_curve=warped_energy.astype(np.float32),
            dtw_path=dtw_path,
            warped_waveform=warped_waveform.astype(np.float32),
            sample_rate=self.target_sr,
        )

    def save_warped_audio(
        self,
        result: ProsodyMappingResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Save the prosody-warped waveform to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), result.warped_waveform, result.sample_rate)
        return path

    def save_mapping(
        self,
        result: ProsodyMappingResult,
        output_path: Union[str, Path],
    ) -> Path:
        """Save DTW path and warped contours as a NumPy archive."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            warped_f0_curve=result.warped_f0_curve,
            warped_energy_curve=result.warped_energy_curve,
            dtw_path=np.asarray(result.dtw_path, dtype=np.int32),
            source_timestamps=result.source_contour.timestamps,
            synthesized_timestamps=result.synthesized_contour.timestamps,
            sample_rate=result.sample_rate,
        )
        return path

    def _compute_dtw_path(
        self,
        source_contour: ProsodyContour,
        synthesized_contour: ProsodyContour,
    ) -> List[Tuple[int, int]]:
        """Run DTW on normalized F0 + energy features."""
        source_features = self._build_feature_matrix(source_contour)
        synth_features = self._build_feature_matrix(synthesized_contour)

        try:
            from fastdtw import fastdtw

            _, path = fastdtw(source_features, synth_features, dist=self._frame_distance)
            return [(int(i), int(j)) for i, j in path]
        except Exception:
            log.warning("fastdtw unavailable or failed; falling back to librosa.sequence.dtw.", exc_info=True)

        cost = self._pairwise_distance(source_features, synth_features)
        _, path = librosa.sequence.dtw(C=cost)
        path = path[::-1]
        return [(int(i), int(j)) for i, j in path]

    def _build_feature_matrix(self, contour: ProsodyContour) -> np.ndarray:
        """Build a 2D feature matrix [normalized log-f0, normalized energy]."""
        f0 = contour.f0_curve.astype(np.float32)
        voiced = f0 > 1.0
        safe_f0 = np.where(voiced, np.log(f0 + 1e-6), 0.0)
        norm_f0 = self._zscore(safe_f0)
        norm_energy = self._zscore(contour.energy_curve.astype(np.float32))
        return np.stack([norm_f0, norm_energy], axis=1)

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-8:
            return np.zeros_like(values)
        return (values - mean) / std

    @staticmethod
    def _frame_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _pairwise_distance(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = source[:, None, :] - target[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _warp_curve(
        self,
        source_curve: np.ndarray,
        target_len: int,
        dtw_path: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        """Project a source contour onto the target timeline using the DTW path."""
        buckets: List[List[float]] = [[] for _ in range(target_len)]
        for src_idx, tgt_idx in dtw_path:
            if 0 <= tgt_idx < target_len and 0 <= src_idx < len(source_curve):
                buckets[tgt_idx].append(float(source_curve[src_idx]))

        warped = np.zeros(target_len, dtype=np.float32)
        last_value = 0.0
        for idx, bucket in enumerate(buckets):
            if bucket:
                last_value = float(np.mean(bucket))
            warped[idx] = last_value

        if target_len > 0 and np.allclose(warped, 0.0):
            warped[:] = float(np.mean(source_curve)) if len(source_curve) else 0.0
        return warped

    def _apply_warping(
        self,
        synth_waveform: np.ndarray,
        synthesized_contour: ProsodyContour,
        warped_f0: np.ndarray,
        warped_energy: np.ndarray,
    ) -> np.ndarray:
        """Apply warped F0 and energy contours to the synthesized waveform."""
        try:
            return self._apply_pyworld_warping(
                synth_waveform=synth_waveform,
                synthesized_contour=synthesized_contour,
                warped_f0=warped_f0,
                warped_energy=warped_energy,
            )
        except Exception:
            log.warning("pyworld-based warping failed; using librosa fallback.", exc_info=True)
            return self._apply_fallback_warping(
                synth_waveform=synth_waveform,
                synthesized_contour=synthesized_contour,
                warped_f0=warped_f0,
                warped_energy=warped_energy,
            )

    def _apply_pyworld_warping(
        self,
        synth_waveform: np.ndarray,
        synthesized_contour: ProsodyContour,
        warped_f0: np.ndarray,
        warped_energy: np.ndarray,
    ) -> np.ndarray:
        """Use WORLD decomposition and resynthesis for contour-level warping."""
        import pyworld as pw

        frame_period_ms = (self.hop_length / self.target_sr) * 1000.0
        f0, time_axis = pw.dio(
            synth_waveform.astype(np.float64),
            fs=self.target_sr,
            f0_floor=self.extractor.f0_floor,
            f0_ceil=self.extractor.f0_ceil,
            frame_period=frame_period_ms,
        )
        f0 = pw.stonemask(synth_waveform.astype(np.float64), f0, time_axis, self.target_sr)
        sp = pw.cheaptrick(synth_waveform.astype(np.float64), f0, time_axis, self.target_sr)
        ap = pw.d4c(synth_waveform.astype(np.float64), f0, time_axis, self.target_sr)

        target_f0 = self._align_length(warped_f0.astype(np.float64), len(f0))
        target_energy = self._align_length(warped_energy.astype(np.float64), len(f0))
        synth_energy = self._align_length(synthesized_contour.energy_curve.astype(np.float64), len(f0))

        voiced = target_f0 > 1.0
        target_f0 = np.where(voiced, target_f0, 0.0)

        gain = target_energy / np.maximum(synth_energy, 1e-6)
        sp = sp * gain[:, None]
        warped = pw.synthesize(target_f0, sp, ap, self.target_sr, frame_period_ms)
        return np.clip(warped, -1.0, 1.0).astype(np.float32)

    def _apply_fallback_warping(
        self,
        synth_waveform: np.ndarray,
        synthesized_contour: ProsodyContour,
        warped_f0: np.ndarray,
        warped_energy: np.ndarray,
    ) -> np.ndarray:
        """Fallback prosody warping using global pitch shift plus framewise energy gain."""
        voiced_src = synthesized_contour.f0_curve[synthesized_contour.f0_curve > 1.0]
        voiced_tgt = warped_f0[warped_f0 > 1.0]

        shifted = synth_waveform.astype(np.float32)
        if len(voiced_src) > 0 and len(voiced_tgt) > 0:
            ratio = float(np.median(voiced_tgt) / max(np.median(voiced_src), 1e-6))
            n_steps = 12.0 * np.log2(max(ratio, 1e-6))
            shifted = librosa.effects.pitch_shift(
                shifted,
                sr=self.target_sr,
                n_steps=n_steps,
            ).astype(np.float32)

        gain_envelope = self._frame_curve_to_sample_curve(
            warped_energy / np.maximum(synthesized_contour.energy_curve, 1e-6),
            num_samples=len(shifted),
        )
        shifted = shifted * gain_envelope
        return np.clip(shifted, -1.0, 1.0).astype(np.float32)

    def _frame_curve_to_sample_curve(self, curve: np.ndarray, num_samples: int) -> np.ndarray:
        """Upsample a frame-level contour to sample-level gains."""
        frame_times = np.arange(len(curve)) * self.hop_length
        sample_times = np.arange(num_samples)
        if len(curve) == 0:
            return np.ones(num_samples, dtype=np.float32)
        return np.interp(sample_times, frame_times, curve, left=curve[0], right=curve[-1]).astype(np.float32)

    @staticmethod
    def _align_length(values: np.ndarray, target_len: int) -> np.ndarray:
        if len(values) == target_len:
            return values
        if len(values) > target_len:
            return values[:target_len]
        if len(values) == 0:
            return np.zeros(target_len, dtype=np.float64)
        return np.pad(values, (0, target_len - len(values)), mode="edge")

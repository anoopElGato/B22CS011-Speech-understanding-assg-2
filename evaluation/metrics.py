"""
evaluation/metrics.py
=====================
Unified evaluation metrics for the speech processing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from jiwer import wer

from spoofing.classifier import compute_eer

log = logging.getLogger(__name__)

AudioInput = Union[str, Path, np.ndarray, torch.Tensor]
SegmentList = Sequence[Dict[str, float | str]]


class Evaluator:
    """
    Compute all project evaluation metrics and return them as a dictionary.

    Metrics
    -------
    - WER using `jiwer`
    - MCD using MFCC alignment with DTW
    - EER for spoof detection
    - LID switching accuracy using duration-weighted timestamp agreement
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        n_mfcc: int = 13,
        frame_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self._mfcc_transform = T.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=self.n_mfcc + 1,
            melkwargs={
                "n_fft": 512,
                "win_length": self.frame_length,
                "hop_length": self.hop_length,
                "n_mels": 40,
                "center": True,
                "power": 2.0,
            },
            log_mels=True,
        )

    def evaluate(
        self,
        reference_text: Optional[str] = None,
        hypothesis_text: Optional[str] = None,
        reference_audio: Optional[AudioInput] = None,
        synthesized_audio: Optional[AudioInput] = None,
        spoof_labels: Optional[Sequence[int]] = None,
        spoof_scores: Optional[Sequence[float]] = None,
        reference_lid_segments: Optional[SegmentList] = None,
        predicted_lid_segments: Optional[SegmentList] = None,
        reference_audio_sr: Optional[int] = None,
        synthesized_audio_sr: Optional[int] = None,
    ) -> Dict[str, Optional[float]]:
        """Return all supported metrics in a single dictionary."""
        metrics: Dict[str, Optional[float]] = {
            "WER": None,
            "MCD": None,
            "EER": None,
            "LID_switching_accuracy": None,
        }

        if reference_text is not None and hypothesis_text is not None:
            metrics["WER"] = self.compute_wer(reference_text, hypothesis_text)

        if reference_audio is not None and synthesized_audio is not None:
            metrics["MCD"] = self.compute_mcd(
                reference_audio=reference_audio,
                synthesized_audio=synthesized_audio,
                reference_audio_sr=reference_audio_sr,
                synthesized_audio_sr=synthesized_audio_sr,
            )

        if spoof_labels is not None and spoof_scores is not None:
            metrics["EER"] = self.compute_eer(spoof_labels, spoof_scores)

        if reference_lid_segments is not None and predicted_lid_segments is not None:
            metrics["LID_switching_accuracy"] = self.compute_lid_switching_accuracy(
                reference_lid_segments=reference_lid_segments,
                predicted_lid_segments=predicted_lid_segments,
            )

        return metrics

    @staticmethod
    def compute_wer(reference_text: str, hypothesis_text: str) -> float:
        """Compute Word Error Rate using jiwer."""
        return float(wer(reference_text, hypothesis_text))

    def compute_mcd(
        self,
        reference_audio: AudioInput,
        synthesized_audio: AudioInput,
        reference_audio_sr: Optional[int] = None,
        synthesized_audio_sr: Optional[int] = None,
    ) -> float:
        """
        Compute Mel-Cepstral Distortion (MCD) with DTW alignment.

        Implementation notes
        --------------------
        - Uses MFCCs as mel-cepstral features.
        - Drops the 0th coefficient and uses coefficients 1..n_mfcc.
        - Aligns frame sequences with DTW before computing frame-wise distortion.
        """
        ref_wav = self._load_audio(reference_audio, reference_audio_sr)
        syn_wav = self._load_audio(synthesized_audio, synthesized_audio_sr)

        ref_mcep = self._extract_mfcc_for_mcd(ref_wav)
        syn_mcep = self._extract_mfcc_for_mcd(syn_wav)

        if ref_mcep.shape[0] == 0 or syn_mcep.shape[0] == 0:
            raise ValueError("Audio is too short to compute MCD.")

        cost = self._pairwise_distance(ref_mcep, syn_mcep)
        path = self._dtw_path(cost)

        aligned_distances = [
            np.linalg.norm(ref_mcep[int(i)] - syn_mcep[int(j)])
            for i, j in path
        ]
        average_distance = float(np.mean(aligned_distances))

        mcd_constant = 10.0 * np.sqrt(2.0) / np.log(10.0)
        return float(mcd_constant * average_distance)

    @staticmethod
    def compute_eer(spoof_labels: Sequence[int], spoof_scores: Sequence[float]) -> float:
        """Compute EER from bona fide/spoof labels and classifier scores."""
        return float(compute_eer(spoof_labels, spoof_scores))

    @staticmethod
    def compute_lid_switching_accuracy(
        reference_lid_segments: SegmentList,
        predicted_lid_segments: SegmentList,
    ) -> float:
        """
        Compute duration-weighted accuracy between timestamped LID segments.

        The metric builds a shared segmentation grid from both reference and
        predicted boundaries, then measures how much time the active language
        label matches between the two timelines.
        """
        if not reference_lid_segments or not predicted_lid_segments:
            raise ValueError("Both reference and predicted LID segments are required.")

        boundaries = sorted(
            {
                float(seg["start"]) for seg in reference_lid_segments
            }
            | {
                float(seg["end"]) for seg in reference_lid_segments
            }
            | {
                float(seg["start"]) for seg in predicted_lid_segments
            }
            | {
                float(seg["end"]) for seg in predicted_lid_segments
            }
        )

        if len(boundaries) < 2:
            raise ValueError("At least one non-zero segment duration is required.")

        matched_duration = 0.0
        total_duration = 0.0

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            duration = right - left
            if duration <= 0:
                continue

            midpoint = (left + right) / 2.0
            ref_lang = Evaluator._label_at_time(reference_lid_segments, midpoint)
            pred_lang = Evaluator._label_at_time(predicted_lid_segments, midpoint)

            if ref_lang is None:
                continue

            total_duration += duration
            if ref_lang == pred_lang:
                matched_duration += duration

        if total_duration <= 0:
            raise ValueError("No valid reference duration found for LID switching accuracy.")

        return float(matched_duration / total_duration)

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

        return waveform.squeeze(0).numpy().astype(np.float32)

    def _extract_mfcc_for_mcd(self, waveform: np.ndarray) -> np.ndarray:
        """Extract mel-cepstral features for MCD."""
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        with torch.no_grad():
            mfcc = self._mfcc_transform(tensor).squeeze(0).cpu().numpy()
        if mfcc.shape[0] <= 1:
            return np.zeros((0, self.n_mfcc), dtype=np.float32)
        return mfcc[1 : self.n_mfcc + 1].T.astype(np.float32)

    @staticmethod
    def _pairwise_distance(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = source[:, None, :] - target[None, :, :]
        return np.linalg.norm(diff, axis=2)

    @staticmethod
    def _dtw_path(cost: np.ndarray) -> List[tuple[int, int]]:
        """Compute a DTW alignment path with simple dynamic programming."""
        n_rows, n_cols = cost.shape
        acc = np.full((n_rows + 1, n_cols + 1), np.inf, dtype=np.float64)
        acc[0, 0] = 0.0

        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(
                    acc[i - 1, j],
                    acc[i, j - 1],
                    acc[i - 1, j - 1],
                )

        i, j = n_rows, n_cols
        path: List[tuple[int, int]] = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (acc[i - 1, j - 1], i - 1, j - 1),
                (acc[i - 1, j], i - 1, j),
                (acc[i, j - 1], i, j - 1),
            ]
            _, next_i, next_j = min(candidates, key=lambda item: item[0])
            i, j = next_i, next_j

        path.reverse()
        return path

    @staticmethod
    def _label_at_time(segments: SegmentList, time_sec: float) -> Optional[str]:
        """Return the active language label at a given timestamp."""
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            if start <= time_sec < end:
                return str(seg["lang"])

        if segments and np.isclose(time_sec, float(segments[-1]["end"])):
            return str(segments[-1]["lang"])
        return None

    def __repr__(self) -> str:
        return (
            f"Evaluator(target_sr={self.target_sr}, n_mfcc={self.n_mfcc}, "
            f"frame_length={self.frame_length}, hop_length={self.hop_length})"
        )

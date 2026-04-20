"""
scripts/find_lid_fgsm_threshold.py
==================================
Find the smallest FGSM epsilon that flips a 5-second Hindi segment to English.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

from lid.language_identifier import FGSMAttackResult, LIDModule


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("find_lid_fgsm_threshold")

TARGET_SR = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the minimum FGSM epsilon that flips Hindi to English on a 5-second segment."
    )
    parser.add_argument("--lecture", required=True, help="Path to lecture audio.")
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds for the 5-second segment.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Segment duration in seconds. Default: 5.0",
    )
    parser.add_argument(
        "--max-epsilon",
        type=float,
        default=0.05,
        help="Upper bound for epsilon search. Default: 0.05",
    )
    parser.add_argument(
        "--min-snr-db",
        type=float,
        default=40.0,
        help="Minimum SNR in dB. Default: 40.0",
    )
    parser.add_argument(
        "--binary-search-steps",
        type=int,
        default=12,
        help="Number of binary-search iterations. Default: 12",
    )
    parser.add_argument(
        "--fgsm-search-steps",
        type=int,
        default=20,
        help="Inner FGSM epsilon grid steps. Default: 20",
    )
    parser.add_argument(
        "--save-audio",
        default=None,
        help="Optional path to save the perturbed segment as WAV.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the result summary as JSON.",
    )
    return parser.parse_args()


def load_segment(path: str, start_sec: float, duration_sec: float) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform)

    start_sample = int(start_sec * TARGET_SR)
    end_sample = start_sample + int(duration_sec * TARGET_SR)
    segment = waveform[:, start_sample:end_sample]

    if segment.shape[1] == 0:
        raise ValueError("Selected segment is empty. Check --start and --duration.")
    if segment.shape[1] < int(duration_sec * TARGET_SR):
        raise ValueError("Selected segment is shorter than the requested duration.")

    return segment.squeeze(0).numpy().astype(np.float32)


def find_threshold(
    lid: LIDModule,
    segment: np.ndarray,
    max_epsilon: float,
    min_snr_db: float,
    binary_search_steps: int,
    fgsm_search_steps: int,
) -> FGSMAttackResult:
    baseline = lid.predict_utterance(segment, source_sr=TARGET_SR)
    log.info(
        "Original prediction: %s | P(hi)=%.4f | P(en)=%.4f",
        baseline["prediction"],
        baseline["prob_hi"],
        baseline["prob_en"],
    )

    if baseline["prediction"] != "hi":
        raise RuntimeError(
            f"Selected segment is not classified as Hindi. Current prediction: {baseline['prediction']}"
        )

    low = 0.0
    high = max_epsilon
    best_result: FGSMAttackResult | None = None

    for step in range(binary_search_steps):
        mid = (low + high) / 2.0
        log.info("Binary search step %d/%d | epsilon=%.8f", step + 1, binary_search_steps, mid)

        try:
            result = lid.fgsm_attack(
                segment,
                source_sr=TARGET_SR,
                epsilon=mid,
                min_snr_db=min_snr_db,
                search_steps=fgsm_search_steps,
            )
            if result.perturbed_prediction == "en":
                best_result = result
                high = result.epsilon_used
            else:
                low = mid
        except RuntimeError:
            low = mid

    if best_result is None:
        raise RuntimeError(
            "Could not flip Hindi to English within the provided epsilon/SNR budget."
        )

    return best_result


def result_to_dict(result: FGSMAttackResult) -> Dict[str, Any]:
    return {
        "epsilon_used": result.epsilon_used,
        "snr_db": result.snr_db,
        "original_prediction": result.original_prediction,
        "perturbed_prediction": result.perturbed_prediction,
        "original_prob_hi": result.original_prob_hi,
        "original_prob_en": result.original_prob_en,
        "perturbed_prob_hi": result.perturbed_prob_hi,
        "perturbed_prob_en": result.perturbed_prob_en,
    }


def main() -> int:
    args = parse_args()

    try:
        segment = load_segment(args.lecture, args.start, args.duration)
        lid = LIDModule()
        result = find_threshold(
            lid=lid,
            segment=segment,
            max_epsilon=args.max_epsilon,
            min_snr_db=args.min_snr_db,
            binary_search_steps=args.binary_search_steps,
            fgsm_search_steps=args.fgsm_search_steps,
        )

        summary = result_to_dict(result)
        print(json.dumps(summary, indent=2))

        if args.save_audio:
            output_audio = Path(args.save_audio)
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_audio), result.perturbed_audio, TARGET_SR)
            log.info("Saved perturbed audio to %s", output_audio)

        if args.save_json:
            output_json = Path(args.save_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log.info("Saved summary JSON to %s", output_json)

        return 0
    except Exception as exc:
        log.exception("Failed to find FGSM threshold: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
pipeline.py
===========
End-to-end CLI pipeline for the code-switched speech processing system.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torchaudio

from evaluation.metrics import Evaluator
from lid.language_identifier import LIDModule
from phonetics.g2p import G2PConverter
from preprocessing.audio_preprocessor import AudioPreprocessor
from prosody.extractor import ProsodyContour, ProsodyExtractor
from prosody.mapper import ProsodyMapper, ProsodyMappingResult
from spoofing.classifier import AntiSpoofClassifier
from spoofing.feature_extractor import AntiSpoofFeatureExtractor
from stt.transcriber import Transcriber, TranscriptionResult
from translation.translator import Translator
from tts.speaker_embedder import SpeakerEmbedder
from tts.synthesiser import PretrainedTTSSynthesiser, SynthesisResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")


@dataclass
class PipelineOutputs:
    """Serializable summary of the pipeline outputs."""

    clean_lecture_audio: str
    transcript_file: str
    ipa_file: str
    translated_file: str
    synthesized_audio: str
    warped_audio: str
    metrics_file: str
    spoof_checkpoint: Optional[str] = None


# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Code-switched speech processing pipeline")
#     parser.add_argument("--lecture", required=True, help="Path to lecture audio")
#     parser.add_argument("--speaker", required=True, help="Path to 60s speaker reference audio")
#     parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
#     return parser

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code-switched speech processing pipeline")
    parser.add_argument("--lecture", required=True, help="Path to lecture audio")
    parser.add_argument("--speaker", required=True, help="Path to 60s speaker reference audio")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument(
        "--tts_backend",
        type=str,
        default="auto",
        choices=["auto", "coqui", "mms"],
        help="TTS backend: 'coqui' for XTTS voice cloning, 'mms' for Meta MMS VITS, or 'auto'.",
    )
    parser.add_argument(
        "--tts_language",
        type=str,
        default="en",
        help="Language code for the TTS backend (for example: en, hi).",
    )
    parser.add_argument(
        "--tts_hf_model",
        type=str,
        default=None,
        help="Optional Hugging Face model name for the MMS/VITS backend.",
    )

    # NEW FLAGS
    parser.add_argument("--resume_from_transcript", action="store_true",
                        help="Skip STT and load transcript from file")
    parser.add_argument("--transcript_file", type=str, default=None,
                        help="Path to precomputed transcript")
    parser.add_argument("--lid_file", type=str, default=None,
                        help="Path to saved LID segments JSON")

    return parser


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_audio_mono(source: Any, target_sr: int = 16_000) -> np.ndarray:
    """Load mono float32 audio from a file path or numpy array."""
    if isinstance(source, np.ndarray):
        audio = source.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        return audio

    waveform, sr = torchaudio.load(str(source))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32)


def build_constrained_decoder() -> Optional[object]:
    """
    Best-effort construction of a constrained decoder.

    Falls back to plain Whisper decoding if the N-gram LM stack is unavailable.
    """
    try:
        from stt.constrained_decoder import ConstrainedDecoder
        from stt.ngram_lm import DEFAULT_CORPUS, NgramLM

        lm = NgramLM(n=3, smoothing="laplace")
        sentences = [line.strip() for line in DEFAULT_CORPUS.splitlines() if line.strip()]
        lm.train_from_sentences(sentences)
        decoder = ConstrainedDecoder(lm=lm, alpha=1.5, beta=0.05, beam_size=3)
        log.info("Constrained decoder initialised with a default domain LM.")
        return decoder
    except Exception as exc:
        log.warning(
            "Constrained decoding could not be initialised; falling back to standard Whisper decoding. (%s)",
            exc,
        )
        return None


def translate_to_lrl(translator: Translator, transcript_text: str, ipa_text: str) -> str:
    """
    Translate transcript into the low-resource target language.

    The current translator module uses transcript text directly; `ipa_text` is
    accepted here so the pipeline preserves the requested stage signature.
    """
    _ = ipa_text
    return translator.translate_text(transcript_text)


def warp_with_precomputed_prosody(
    mapper: ProsodyMapper,
    synthesized_audio: np.ndarray,
    synthesized_sr: int,
    src_prosody: ProsodyContour,
    tgt_prosody: ProsodyContour,
) -> ProsodyMappingResult:
    """Apply DTW-based prosody warping using pre-extracted contours."""
    synth_waveform = mapper.extractor._load_audio(synthesized_audio, synthesized_sr).astype(np.float32)
    dtw_path = mapper._compute_dtw_path(src_prosody, tgt_prosody)
    warped_f0 = mapper._warp_curve(src_prosody.f0_curve, len(tgt_prosody.f0_curve), dtw_path)
    warped_energy = mapper._warp_curve(src_prosody.energy_curve, len(tgt_prosody.energy_curve), dtw_path)
    warped_waveform = mapper._apply_warping(
        synth_waveform=synth_waveform,
        synthesized_contour=tgt_prosody,
        warped_f0=warped_f0,
        warped_energy=warped_energy,
    )
    return ProsodyMappingResult(
        source_contour=src_prosody,
        synthesized_contour=tgt_prosody,
        warped_f0_curve=warped_f0.astype(np.float32),
        warped_energy_curve=warped_energy.astype(np.float32),
        dtw_path=dtw_path,
        warped_waveform=warped_waveform.astype(np.float32),
        sample_rate=mapper.target_sr,
    )


def chunk_audio(
    waveform: np.ndarray,
    sr: int,
    chunk_sec: float = 2.0,
    hop_sec: float = 1.0,
) -> List[np.ndarray]:
    """Split audio into overlapping chunks for spoof classifier training."""
    chunk_size = max(1, int(chunk_sec * sr))
    hop_size = max(1, int(hop_sec * sr))

    if len(waveform) <= chunk_size:
        return [waveform.astype(np.float32)]

    chunks: List[np.ndarray] = []
    for start in range(0, max(len(waveform) - chunk_size + 1, 1), hop_size):
        end = start + chunk_size
        if end > len(waveform):
            break
        chunks.append(waveform[start:end].astype(np.float32))

    if not chunks:
        chunks.append(waveform.astype(np.float32))
    return chunks


def split_train_val(
    features: Sequence[np.ndarray],
    labels: Sequence[int],
    val_ratio: float = 0.25,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """Simple class-balanced split for spoof classifier training."""
    real = [feat for feat, label in zip(features, labels) if label == 1]
    spoof = [feat for feat, label in zip(features, labels) if label == 0]

    if not real or not spoof:
        raise ValueError("Need both bona fide and spoof samples to train the anti-spoof classifier.")

    def _split_class(items: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if len(items) == 1:
            return items, items
        split_idx = max(1, int(round(len(items) * (1.0 - val_ratio))))
        split_idx = min(split_idx, len(items) - 1)
        return items[:split_idx], items[split_idx:]

    real_train, real_val = _split_class(real)
    spoof_train, spoof_val = _split_class(spoof)

    train_features = real_train + spoof_train
    train_labels = [1] * len(real_train) + [0] * len(spoof_train)
    val_features = real_val + spoof_val
    val_labels = [1] * len(real_val) + [0] * len(spoof_val)
    return train_features, train_labels, val_features, val_labels


def train_and_evaluate_spoof_classifier(
    clean_lecture_path: Path,
    synthesized_audio: np.ndarray,
    synthesized_sr: int,
    warped_audio: np.ndarray,
    warped_sr: int,
    output_dir: Path,
) -> Tuple[Optional[AntiSpoofClassifier], Dict[str, Optional[float]], Optional[Path], List[int], List[float]]:
    """
    Train and evaluate a binary anti-spoof classifier using available audio.

    Bona fide samples come from the real lecture audio.
    Spoof samples come from synthesized and prosody-warped outputs.
    """
    log.info("Preparing anti-spoofing training data.")
    extractor = AntiSpoofFeatureExtractor(target_sr=16_000)

    real_waveform = load_audio_mono(clean_lecture_path, target_sr=16_000)
    synth_waveform = load_audio_mono(synthesized_audio, target_sr=synthesized_sr)
    warped_waveform = load_audio_mono(warped_audio, target_sr=warped_sr)

    real_chunks = chunk_audio(real_waveform, 16_000)
    spoof_chunks = chunk_audio(synth_waveform, 16_000) + chunk_audio(warped_waveform, 16_000)

    features: List[np.ndarray] = []
    labels: List[int] = []

    for chunk in real_chunks:
        features.append(extractor.extract(chunk, feature_type="lfcc", source_sr=16_000))
        labels.append(1)

    for chunk in spoof_chunks:
        features.append(extractor.extract(chunk, feature_type="lfcc", source_sr=16_000))
        labels.append(0)

    train_features, train_labels, val_features, val_labels = split_train_val(features, labels)

    classifier = AntiSpoofClassifier(model_type="cnn", device="cpu")
    classifier.fit(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        epochs=5,
        batch_size=8,
    )
    eval_metrics = classifier.evaluate(val_features, val_labels, batch_size=8)
    val_scores = classifier.predict_proba(val_features, batch_size=8).tolist()

    checkpoint_path = output_dir / "spoof_classifier.pt"
    classifier.save(checkpoint_path)
    log.info(
        "Anti-spoof classifier trained. Accuracy=%.4f | EER=%.4f",
        eval_metrics["accuracy"],
        eval_metrics["eer"],
    )
    return classifier, eval_metrics, checkpoint_path, val_labels, val_scores


def run_pipeline(lecture_path: str, speaker_path: str, output_dir: str, resume_from_transcript, transcript_file, lid_file, args) -> PipelineOutputs:
    out_dir = ensure_output_dir(output_dir)

    lid_path = out_dir / "lid_segments.json"
    transcript_path = out_dir / "transcript.txt"
    ipa_path = out_dir / "ipa.txt"
    translated_path = out_dir / "translated.txt"
    synthesized_path = out_dir / "synthesized.wav"
    warped_audio_path = out_dir / "output_LRL_cloned.wav"
    metrics_path = out_dir / "metrics.json"

    log.info("Step 1/12 - Preprocessing lecture audio")
    clean_lecture_path = AudioPreprocessor.process(
        input_path=lecture_path,
        output_path=str(out_dir / "lecture_clean.wav"),
        target_sr=16_000,
        backend="spectral",
    )

    # log.info("Step 2/12 - Frame-level language identification")
    # lid_module = LIDModule()
    # lid_segments = lid_module.predict(str(clean_lecture_path))
    # write_json(lid_path, {"segments": lid_segments})

    # log.info("Step 3/12 - STT with constrained decoding")
    constrained_decoder = build_constrained_decoder()
    stt_module = Transcriber(constrained_decoder=constrained_decoder)
    # transcript_result: TranscriptionResult = stt_module.transcribe(str(clean_lecture_path), lid_segments)
    # transcript_text = transcript_result.full_text
    # write_text(transcript_path, transcript_text)

    lid_module = LIDModule()

    if resume_from_transcript:
        log.info("Skipping LID + STT (resuming from transcript)")

        if transcript_file is None:
            raise ValueError("Must provide --transcript_file when using --resume_from_transcript")

        # Load transcript
        transcript_text = Path(transcript_file).read_text(encoding="utf-8")
        write_text(transcript_path, transcript_text)

        # Load or recompute LID
        if lid_file is not None and Path(lid_file).exists():
            log.info("Loading LID segments from file")
            lid_segments = json.loads(Path(lid_file).read_text())["segments"]
        else:
            log.warning("LID file not provided. Recomputing LID.")
            lid_segments = lid_module.predict(str(clean_lecture_path))
            write_json(lid_path, {"segments": lid_segments})

    else:
        log.info("Step 2/12 - Frame-level language identification")
        lid_segments = lid_module.predict(str(clean_lecture_path))

        # Save LID for reuse
        write_json(lid_path, {"segments": lid_segments})

        log.info("Step 3/12 - STT with constrained decoding")
        constrained_decoder = build_constrained_decoder()
        stt_module = Transcriber(constrained_decoder=constrained_decoder)

        transcript_result: TranscriptionResult = stt_module.transcribe(
            str(clean_lecture_path), lid_segments
        )
        transcript_text = transcript_result.full_text
        write_text(transcript_path, transcript_text)

    log.info("Step 4/12 - Transcript to IPA conversion")
    ipa_module = G2PConverter()
    ipa_text = ipa_module.convert(transcript_text)
    write_text(ipa_path, ipa_text)

    log.info("Step 5/12 - Translation to low-resource language")
    translator = Translator()
    translated_text = translate_to_lrl(translator, transcript_text, ipa_text)
    write_text(translated_path, translated_text)

    tts_module = PretrainedTTSSynthesiser(
        backend=args.tts_backend,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        hf_model_name=args.tts_hf_model,
        default_language=args.tts_language,
    )
    speaker_embedding = None
    if tts_module.requires_speaker_embedding:
        log.info("Step 6/12 - Speaker embedding extraction")
        speaker_encoder = SpeakerEmbedder(backend="resemblyzer")
        speaker_embedding = speaker_encoder.extract_embedding(speaker_path)
    else:
        log.info(
            "Step 6/12 - Speaker embedding extraction skipped because TTS backend '%s' does not use voice cloning inputs",
            tts_module.backend,
        )

    log.info("Step 7/12 - TTS synthesis")
    synthesis_result: SynthesisResult = tts_module.synthesize(
        text=translated_text,
        speaker_embedding=speaker_embedding,
        reference_audio=speaker_path,
        language=args.tts_language,
        file_path=synthesized_path,
    )

    log.info("Step 8/12 - Prosody extraction from lecture audio")
    prosody_module = ProsodyExtractor(target_sr=16_000, f0_backend="pyworld")
    src_prosody = prosody_module.extract(str(clean_lecture_path))

    log.info("Step 9/12 - Prosody extraction from synthesized audio")
    tgt_prosody = prosody_module.extract(
        synthesis_result.waveform,
        source_sr=synthesis_result.sample_rate,
    )

    log.info("Step 10/12 - DTW-based prosody warping")
    dtw_aligner = ProsodyMapper(extractor=prosody_module)
    prosody_result = warp_with_precomputed_prosody(
        mapper=dtw_aligner,
        synthesized_audio=synthesis_result.waveform,
        synthesized_sr=synthesis_result.sample_rate,
        src_prosody=src_prosody,
        tgt_prosody=tgt_prosody,
    )
    dtw_aligner.save_mapping(prosody_result, out_dir / "prosody_mapping.npz")
    dtw_aligner.save_warped_audio(prosody_result, warped_audio_path)

    log.info("Step 11/12 - Train and evaluate anti-spoofing classifier")
    spoof_classifier, spoof_eval_metrics, spoof_checkpoint, spoof_val_labels, spoof_val_scores = train_and_evaluate_spoof_classifier(
        clean_lecture_path=clean_lecture_path,
        synthesized_audio=synthesis_result.waveform,
        synthesized_sr=synthesis_result.sample_rate,
        warped_audio=prosody_result.warped_waveform,
        warped_sr=prosody_result.sample_rate,
        output_dir=out_dir,
    )
    _ = spoof_classifier

    log.info("Step 12/12 - Compute evaluation metrics")
    evaluator = Evaluator(target_sr=16_000)

    try:
        warped_lid_segments = lid_module.predict(prosody_result.warped_waveform, source_sr=prosody_result.sample_rate)
    except Exception as exc:
        log.warning("Could not compute LID on warped audio for switching accuracy: %s", exc)
        warped_lid_segments = None

    try:
        warped_transcript_result = stt_module.transcribe(
            prosody_result.warped_waveform,
            warped_lid_segments or lid_segments,
            default_lang="hi",
        )
        warped_transcript_text = warped_transcript_result.full_text
    except Exception as exc:
        log.warning("Could not compute transcript for warped audio WER: %s", exc)
        warped_transcript_text = None

    metrics = evaluator.evaluate(
        reference_text=transcript_text if warped_transcript_text is not None else None,
        hypothesis_text=warped_transcript_text,
        reference_audio=str(clean_lecture_path),
        synthesized_audio=prosody_result.warped_waveform,
        spoof_labels=spoof_val_labels,
        spoof_scores=spoof_val_scores,
        reference_lid_segments=lid_segments if warped_lid_segments is not None else None,
        predicted_lid_segments=warped_lid_segments,
        synthesized_audio_sr=prosody_result.sample_rate,
    )

    metrics["anti_spoof_accuracy"] = spoof_eval_metrics.get("accuracy")
    metrics["anti_spoof_validation_loss"] = spoof_eval_metrics.get("loss")
    write_json(metrics_path, metrics)

    log.info("Pipeline completed successfully.")
    return PipelineOutputs(
        clean_lecture_audio=str(clean_lecture_path),
        transcript_file=str(transcript_path),
        ipa_file=str(ipa_path),
        translated_file=str(translated_path),
        synthesized_audio=str(synthesized_path),
        warped_audio=str(warped_audio_path),
        metrics_file=str(metrics_path),
        spoof_checkpoint=str(spoof_checkpoint) if spoof_checkpoint is not None else None,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        # outputs = run_pipeline(
        #     lecture_path=args.lecture,
        #     speaker_path=args.speaker,
        #     output_dir=args.output_dir,
        #     args= args
        # )
        outputs = run_pipeline(
            lecture_path=args.lecture,
            speaker_path=args.speaker,
            output_dir=args.output_dir,
            resume_from_transcript=args.resume_from_transcript,
            transcript_file=args.transcript_file,
            lid_file=args.lid_file,
            args= args,
        )
        log.info("Saved outputs:\n%s", json.dumps(asdict(outputs), indent=2))
        return 0
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

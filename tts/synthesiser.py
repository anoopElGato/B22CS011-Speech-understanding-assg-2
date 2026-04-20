"""
tts/synthesiser.py
==================
Pretrained TTS integration with batch inference and waveform saving.
"""

from __future__ import annotations

import logging
import os
import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio

log = logging.getLogger(__name__)

TextBatch = Union[str, Sequence[str]]


@dataclass
class SynthesisResult:
    """Container for synthesized speech."""

    text: str
    waveform: np.ndarray
    sample_rate: int
    file_path: Optional[Path] = None


class PretrainedTTSSynthesiser:
    """
    Synthesize speech from translated text using a pretrained XTTS model.

    Notes
    -----
    XTTS-style models accept speaker embeddings for the decoder, but they also
    require GPT conditioning latents derived from reference audio. This class
    therefore takes both a cached `speaker_embedding` and the `reference_audio`
    used to compute the text-conditioning latents.
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        hf_model_name: Optional[str] = None,
        device: Optional[str] = None,
        default_language: str = "en",
        output_sample_rate: int = 24_000,
        temperature: float = 0.7,
        length_penalty: float = 1.0,
        repetition_penalty: float = 2.0,
        top_k: int = 50,
        top_p: float = 0.8,
        speed: float = 1.0,
        enable_text_splitting: bool = True,
    ) -> None:
        self.backend = self._resolve_backend(backend)
        self.model_name = model_name
        self.hf_model_name = hf_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_language = default_language
        self.output_sample_rate = output_sample_rate
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.speed = speed
        self.enable_text_splitting = enable_text_splitting

        self._api = None
        self._xtts_model = None
        self._hf_model = None
        self._hf_tokenizer = None

    def synthesize(
        self,
        text: str,
        speaker_embedding: Union[np.ndarray, torch.Tensor],
        reference_audio: Union[str, Path],
        language: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        conditioning_latents: Optional[torch.Tensor] = None,
        fallback_speaker_embedding: Optional[torch.Tensor] = None,
    ) -> SynthesisResult:
        """Synthesize a single utterance and optionally save it."""
        lang = language or self.default_language
        if self.backend == "coqui":
            model = self._load_model()
            gpt_cond_latent, inferred_speaker_embedding = self._resolve_conditioning(
                reference_audio=reference_audio,
                conditioning_latents=conditioning_latents,
                fallback_speaker_embedding=fallback_speaker_embedding,
                model=model,
            )
            prepared_embedding = self._prepare_speaker_embedding(
                speaker_embedding=speaker_embedding,
                fallback_embedding=inferred_speaker_embedding,
            )

            output = model.inference(
                text,
                lang,
                gpt_cond_latent.to(self.device),
                prepared_embedding.to(self.device),
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
                speed=self.speed,
                enable_text_splitting=self.enable_text_splitting,
            )
            waveform = np.asarray(output["wav"], dtype=np.float32)
        else:
            waveform = self._synthesize_with_mms(
                text=text,
                language=lang,
                speaker_embedding=speaker_embedding,
                reference_audio=reference_audio,
            )

        save_path = Path(file_path) if file_path is not None else None
        if save_path is not None:
            self.save_waveform(waveform, save_path)

        return SynthesisResult(
            text=text,
            waveform=waveform,
            sample_rate=self.output_sample_rate,
            file_path=save_path,
        )

    def synthesize_batch(
        self,
        texts: TextBatch,
        speaker_embedding: Union[np.ndarray, torch.Tensor],
        reference_audio: Union[str, Path],
        language: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: str = "tts_batch",
    ) -> List[SynthesisResult]:
        """Run batch inference over multiple text inputs."""
        items = [texts] if isinstance(texts, str) else list(texts)
        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        gpt_cond_latent = None
        inferred_speaker_embedding = None
        prepared_embedding = speaker_embedding
        if self.backend == "coqui":
            model = self._load_model()
            gpt_cond_latent, inferred_speaker_embedding = self._resolve_conditioning(
                reference_audio=reference_audio,
                conditioning_latents=None,
                fallback_speaker_embedding=None,
                model=model,
            )
            prepared_embedding = self._prepare_speaker_embedding(
                speaker_embedding=speaker_embedding,
                fallback_embedding=inferred_speaker_embedding,
            )

        results: List[SynthesisResult] = []
        for idx, text in enumerate(items):
            save_path = out_dir / f"{filename_prefix}_{idx:03d}.wav" if out_dir else None
            results.append(
                self.synthesize(
                    text=text,
                    speaker_embedding=prepared_embedding,
                    reference_audio=reference_audio,
                    language=language,
                    file_path=save_path,
                    conditioning_latents=gpt_cond_latent,
                    fallback_speaker_embedding=inferred_speaker_embedding,
                )
            )
        return results

    @property
    def requires_speaker_embedding(self) -> bool:
        """Whether the active backend expects voice-cloning conditioning inputs."""
        return self.backend == "coqui"

    def save_waveform(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        file_path: Union[str, Path],
    ) -> Path:
        """Save the final synthesized waveform to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(str(path), tensor.cpu(), self.output_sample_rate)
        return path

    def _prepare_speaker_embedding(
        self,
        speaker_embedding: Union[np.ndarray, torch.Tensor],
        fallback_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize speaker embedding shape to XTTS expected (1, D, 1)."""
        if speaker_embedding is None:
            return fallback_embedding

        if isinstance(speaker_embedding, np.ndarray):
            tensor = torch.from_numpy(speaker_embedding).float()
        else:
            tensor = speaker_embedding.detach().float().cpu()

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        elif tensor.ndim != 3:
            raise ValueError(
                "speaker_embedding must be 1D, 2D, or 3D tensor/array compatible with XTTS."
            )

        expected_dim = fallback_embedding.shape[1]
        if tensor.shape[1] != expected_dim:
            log.warning(
                "Speaker embedding dim %d does not match model dim %d; using model-derived embedding instead.",
                tensor.shape[1],
                expected_dim,
            )
            return fallback_embedding

        return tensor

    def _resolve_conditioning(
        self,
        reference_audio: Union[str, Path],
        conditioning_latents: Optional[torch.Tensor],
        fallback_speaker_embedding: Optional[torch.Tensor],
        model,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve GPT conditioning latents and fallback speaker embedding."""
        if conditioning_latents is not None and fallback_speaker_embedding is not None:
            return conditioning_latents, fallback_speaker_embedding

        return model.get_conditioning_latents(audio_path=[str(reference_audio)])

    def _load_model(self):
        """Load the configured backend model."""
        if self.backend == "coqui":
            return self._load_coqui_model()
        if self.backend == "mms":
            return self._load_mms_model()
        raise ValueError(f"Unsupported TTS backend: {self.backend}")

    def _load_coqui_model(self):
        """Load XTTS through Coqui TTS API and expose the underlying model."""
        if self._xtts_model is not None:
            return self._xtts_model

        try:
            from TTS.api import TTS
        except Exception as exc:
            raise ImportError(self._format_coqui_error(exc)) from exc

        try:
            self._api = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
            self._xtts_model = self._api.synthesizer.tts_model
        except Exception as exc:
            raise RuntimeError(self._format_coqui_error(exc)) from exc

        api_sample_rate = getattr(self._api.synthesizer, "output_sample_rate", None)
        model_audio_cfg = getattr(getattr(self._xtts_model, "config", None), "audio", None)
        model_sample_rate = getattr(model_audio_cfg, "output_sample_rate", None)
        self.output_sample_rate = int(api_sample_rate or model_sample_rate or self.output_sample_rate)

        return self._xtts_model

    def _load_mms_model(self) -> Tuple[Any, Any]:
        """Load a Meta MMS VITS-style model through Hugging Face Transformers."""
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return self._hf_model, self._hf_tokenizer

        model_name = self.hf_model_name or self._default_mms_model(self.default_language)
        try:
            from transformers import AutoTokenizer, VitsModel
        except Exception as exc:
            raise ImportError(
                "Transformers VITS backend is unavailable. Install transformers and its audio dependencies."
            ) from exc

        try:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = VitsModel.from_pretrained(model_name).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load MMS/VITS model '{model_name}'. If it is not cached locally, "
                "download it in an environment with internet access or choose another backend."
            ) from exc

        sample_rate = getattr(getattr(self._hf_model, "config", None), "sampling_rate", None)
        if sample_rate is not None:
            self.output_sample_rate = int(sample_rate)
        return self._hf_model, self._hf_tokenizer

    def _synthesize_with_mms(
        self,
        text: str,
        language: Optional[str],
        speaker_embedding: Union[np.ndarray, torch.Tensor, None],
        reference_audio: Union[str, Path, None],
    ) -> np.ndarray:
        """
        Generate MMS/VITS audio.

        When mock objects are injected for tests, inference runs in-process.
        Otherwise, it runs in a clean subprocess to avoid SpeechBrain lazy-import
        side effects contaminating the transformers import path.
        """
        if speaker_embedding is not None or reference_audio is not None:
            log.info(
                "TTS backend '%s' ignores speaker cloning inputs and uses a pretrained generative voice.",
                self.backend,
            )
        resolved_model_name = self._resolve_mms_model_name(text=text, language=language)

        if self._hf_model is not None and self._hf_tokenizer is not None:
            self._hf_model.eval()
            waveforms: List[np.ndarray] = []
            chunks = self._split_text_for_mms(text)
            if not chunks:
                raise ValueError("MMS synthesis received empty text after normalization/chunking.")
            for chunk in chunks:
                inputs = self._hf_tokenizer(text=chunk, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():
                    output = self._hf_model(**inputs)
                waveforms.append(output.waveform.squeeze().detach().cpu().numpy().astype(np.float32))
            return self._concat_waveforms(waveforms)

        return self._synthesize_with_mms_subprocess(text, resolved_model_name)

    def _synthesize_with_mms_subprocess(self, text: str, model_name: Optional[str] = None) -> np.ndarray:
        """Run MMS inference in a clean Python subprocess."""
        model_name = model_name or self.hf_model_name or self._default_mms_model(self.default_language)
        temp_root = Path.cwd() / "outputs" / "_tmp_mms_tts"
        temp_root.mkdir(parents=True, exist_ok=True)

        script = """
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

request = json.loads(Path(__import__("sys").argv[1]).read_text(encoding="utf-8"))
model_name = request["model_name"]
texts = request["texts"]
device = request["device"]
output_path = request["output_path"]
gap_samples = int(request.get("gap_samples", 0))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VitsModel.from_pretrained(model_name).to(device)
model.eval()
pieces = []
for index, text in enumerate(texts):
    inputs = tokenizer(text=text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().detach().cpu().numpy().astype(np.float32)
    pieces.append(waveform)
    if gap_samples > 0 and index < len(texts) - 1:
        pieces.append(np.zeros(gap_samples, dtype=np.float32))

waveform = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

np.save(output_path, waveform)
print(json.dumps({"sample_rate": int(getattr(model.config, "sampling_rate", 16000))}))
"""

        work_dir = temp_root / f"job_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_path = work_dir / "waveform.npy"
            request_path = work_dir / "request.json"
            chunks = self._split_text_for_mms(text)
            if not chunks:
                raise ValueError("MMS synthesis received empty text after normalization/chunking.")
            preferred_device = self._preferred_mms_device()
            request_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "texts": chunks,
                        "device": preferred_device,
                        "output_path": str(out_path),
                        "gap_samples": int(self.output_sample_rate * 0.15),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = self._run_mms_subprocess(script, request_path)
            if result.returncode != 0 and preferred_device == "cuda":
                details = (result.stderr or result.stdout or "").strip().lower()
                if "outofmemory" in details or "cuda out of memory" in details:
                    log.warning("MMS synthesis ran out of CUDA memory; retrying on CPU.")
                    request_path.write_text(
                        json.dumps(
                            {
                                "model_name": model_name,
                                "texts": chunks,
                                "device": "cpu",
                                "output_path": str(out_path),
                                "gap_samples": int(self.output_sample_rate * 0.15),
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    result = self._run_mms_subprocess(script, request_path)

            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"Could not run MMS/VITS model '{model_name}' in a clean subprocess. {details}"
                )
            if not out_path.exists():
                raise RuntimeError("MMS/VITS subprocess completed without producing audio output.")

            payload = json.loads(result.stdout.strip().splitlines()[-1])
            self.output_sample_rate = int(payload.get("sample_rate", self.output_sample_rate))
            waveform = np.load(out_path)
            return waveform.astype(np.float32)
        finally:
            for path in sorted(work_dir.glob("*"), reverse=True):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                work_dir.rmdir()
            except OSError:
                pass

    def _run_mms_subprocess(self, script: str, request_path: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        return subprocess.run(
            [sys.executable, "-c", script, str(request_path)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def _preferred_mms_device(self) -> str:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return self.device
        return "cpu"

    def _resolve_mms_model_name(self, text: str, language: Optional[str]) -> str:
        if self.hf_model_name:
            return self.hf_model_name

        script = self._detect_text_script(text)
        if script == "gujarati":
            return self._default_mms_model("guj")
        if script == "devanagari":
            return self._default_mms_model("hin")
        return self._default_mms_model(language or self.default_language)

    def _detect_text_script(self, text: str) -> str:
        counts = {"gujarati": 0, "devanagari": 0, "latin": 0}
        for ch in text:
            code = ord(ch)
            if 0x0A80 <= code <= 0x0AFF:
                counts["gujarati"] += 1
            elif 0x0900 <= code <= 0x097F:
                counts["devanagari"] += 1
            elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                counts["latin"] += 1
        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 0 else "unknown"

    def _split_text_for_mms(self, text: str, max_chars: int = 220) -> List[str]:
        """Split long text into smaller chunks to keep VITS attention memory bounded."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        if not self.enable_text_splitting or len(normalized) <= max_chars:
            return [normalized] if self._is_valid_mms_chunk(normalized) else []

        sentences = re.split(r"(?<=[.!?;:])\s+", normalized)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(sentence) <= max_chars:
                current = sentence
                continue

            words = sentence.split()
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = word
            if current:
                chunks.append(current)
                current = ""

        if current:
            chunks.append(current)

        cleaned = [chunk.strip() for chunk in chunks if self._is_valid_mms_chunk(chunk)]
        if cleaned:
            return cleaned
        return [normalized] if self._is_valid_mms_chunk(normalized) else []

    def _concat_waveforms(self, waveforms: Sequence[np.ndarray], gap_ms: int = 150) -> np.ndarray:
        if not waveforms:
            return np.zeros(0, dtype=np.float32)
        if len(waveforms) == 1:
            return np.asarray(waveforms[0], dtype=np.float32)
        gap = np.zeros(int(self.output_sample_rate * gap_ms / 1000.0), dtype=np.float32)
        pieces: List[np.ndarray] = []
        for index, waveform in enumerate(waveforms):
            pieces.append(np.asarray(waveform, dtype=np.float32))
            if index < len(waveforms) - 1:
                pieces.append(gap)
        return np.concatenate(pieces).astype(np.float32)

    def _is_valid_mms_chunk(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return bool(re.search(r"[A-Za-z0-9\u0900-\u097F]", stripped))

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "coqui" if sys.version_info < (3, 12) else "mms"
        return backend

    def _default_mms_model(self, language: str) -> str:
        model_map: Dict[str, str] = {
            "en": "facebook/mms-tts-eng",
            "eng": "facebook/mms-tts-eng",
            "hi": "facebook/mms-tts-hin",
            "hin": "facebook/mms-tts-hin",
            "gu": "facebook/mms-tts-guj",
            "guj": "facebook/mms-tts-guj",
        }
        return model_map.get(language.lower(), "facebook/mms-tts-eng")

    def _format_coqui_error(self, exc: Exception) -> str:
        base = (
            "Coqui XTTS could not be loaded. "
            "Coqui TTS/XTTS is not reliably supported on Python 3.12+; use Python 3.11 for XTTS/YourTTS "
            "or switch this project to the 'mms' backend on newer Python versions."
        )
        details = str(exc).strip()
        if details:
            return f"{base} Original error: {details}"
        return base

    def __repr__(self) -> str:
        return (
            f"PretrainedTTSSynthesiser(backend={self.backend!r}, model_name={self.model_name!r}, "
            f"hf_model_name={self.hf_model_name!r}, device={self.device!r}, "
            f"sample_rate={self.output_sample_rate})"
        )

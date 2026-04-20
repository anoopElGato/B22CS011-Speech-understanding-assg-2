"""TTS utilities package."""

from .speaker_embedder import SpeakerEmbedder
from .synthesiser import PretrainedTTSSynthesiser

__all__ = ["SpeakerEmbedder", "PretrainedTTSSynthesiser"]

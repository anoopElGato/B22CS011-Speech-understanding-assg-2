"""Spoofing utilities package."""

from .classifier import AntiSpoofClassifier, compute_eer
from .feature_extractor import AntiSpoofFeatureExtractor

__all__ = ["AntiSpoofFeatureExtractor", "AntiSpoofClassifier", "compute_eer"]

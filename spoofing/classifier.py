"""
spoofing/classifier.py
======================
Binary anti-spoof classifier for bona fide vs spoofed speech.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

FeatureInput = Union[np.ndarray, torch.Tensor]
ModelType = Literal["cnn", "cnn_lstm"]


@dataclass
class SpoofSample:
    """Single training or inference example."""

    features: np.ndarray
    label: int


@dataclass
class TrainingHistory:
    """Tracked loss/metric history across epochs."""

    train_loss: List[float]
    val_loss: List[float]
    val_eer: List[float]


def compute_eer(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    Compute equal error rate for bona fide/spoof scores.

    Assumes score = P(bona fide), with labels:
    - 1: bona fide / real
    - 0: spoof
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if len(labels) == 0:
        raise ValueError("labels and scores must be non-empty.")

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


class FeatureDataset(Dataset[SpoofSample]):
    """Dataset wrapper for precomputed LFCC/CQCC feature matrices."""

    def __init__(self, samples: Sequence[SpoofSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SpoofSample:
        return self.samples[index]


def collate_feature_batch(batch: Sequence[SpoofSample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length feature matrices to a common frame count.

    Returns
    -------
    features : (batch, 1, max_frames, num_coeffs)
    lengths  : (batch,)
    labels   : (batch,)
    """
    if not batch:
        raise ValueError("Batch must not be empty.")

    num_coeffs = batch[0].features.shape[1]
    max_frames = max(sample.features.shape[0] for sample in batch)

    features = torch.zeros(len(batch), 1, max_frames, num_coeffs, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.float32)

    for idx, sample in enumerate(batch):
        matrix = torch.as_tensor(sample.features, dtype=torch.float32)
        frames = matrix.shape[0]
        features[idx, 0, :frames, :] = matrix
        lengths[idx] = frames
        labels[idx] = float(sample.label)

    return features, lengths, labels


class CNNBackbone(nn.Module):
    """Compact CNN encoder over time-frequency spoofing features."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        return self.projection(encoded)


class CNNLSTMBackbone(nn.Module):
    """CNN front end followed by temporal LSTM pooling."""

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.cnn(features)
        encoded = encoded.mean(dim=3)
        encoded = encoded.transpose(1, 2)
        _, (hidden, _) = self.lstm(encoded)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.output(pooled)


class AntiSpoofNet(nn.Module):
    """Binary classifier head over spoofing feature embeddings."""

    def __init__(self, model_type: ModelType = "cnn", embedding_dim: int = 128) -> None:
        super().__init__()
        if model_type == "cnn":
            self.backbone = CNNBackbone(embedding_dim=embedding_dim)
        elif model_type == "cnn_lstm":
            self.backbone = CNNLSTMBackbone(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(features)
        logits = self.classifier(embedding).squeeze(-1)
        return logits


class AntiSpoofClassifier:
    """
    Trainable bona fide vs spoof classifier using LFCC/CQCC features.

    Labels:
    - `1`: bona fide / real
    - `0`: spoof
    """

    def __init__(
        self,
        model_type: ModelType = "cnn",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ) -> None:
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AntiSpoofNet(model_type=model_type).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def fit(
        self,
        train_features: Sequence[FeatureInput],
        train_labels: Sequence[int],
        val_features: Optional[Sequence[FeatureInput]] = None,
        val_labels: Optional[Sequence[int]] = None,
        epochs: int = 10,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> TrainingHistory:
        """Train the classifier and optionally track validation EER."""
        train_loader = self._build_loader(train_features, train_labels, batch_size=batch_size, shuffle=shuffle)

        history = TrainingHistory(train_loss=[], val_loss=[], val_eer=[])

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            total_examples = 0

            for features, _lengths, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

                batch_size_actual = labels.shape[0]
                running_loss += float(loss.item()) * batch_size_actual
                total_examples += batch_size_actual

            epoch_train_loss = running_loss / max(total_examples, 1)
            history.train_loss.append(epoch_train_loss)

            if val_features is not None and val_labels is not None:
                metrics = self.evaluate(val_features, val_labels, batch_size=batch_size)
                history.val_loss.append(metrics["loss"])
                history.val_eer.append(metrics["eer"])
                log.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_eer=%.4f",
                    epoch + 1,
                    epochs,
                    epoch_train_loss,
                    metrics["loss"],
                    metrics["eer"],
                )
            else:
                log.info(
                    "Epoch %d/%d | train_loss=%.4f",
                    epoch + 1,
                    epochs,
                    epoch_train_loss,
                )

        return history

    @torch.no_grad()
    def predict_proba(
        self,
        features: Sequence[FeatureInput],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return P(bona fide) for each feature matrix."""
        dummy_labels = [0] * len(features)
        loader = self._build_loader(features, dummy_labels, batch_size=batch_size, shuffle=False)

        self.model.eval()
        probs: List[np.ndarray] = []
        for batch_features, _lengths, _labels in loader:
            batch_features = batch_features.to(self.device)
            logits = self.model(batch_features)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(batch_probs.astype(np.float32))

        return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)

    @torch.no_grad()
    def predict(
        self,
        features: Sequence[FeatureInput],
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> List[str]:
        """Return hard labels as `bona_fide` or `spoof`."""
        probs = self.predict_proba(features, batch_size=batch_size)
        return ["bona_fide" if prob >= threshold else "spoof" for prob in probs]

    @torch.no_grad()
    def evaluate(
        self,
        features: Sequence[FeatureInput],
        labels: Sequence[int],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Compute loss, accuracy, and EER on a validation/test split."""
        loader = self._build_loader(features, labels, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        all_probs: List[float] = []
        all_labels: List[int] = []

        for batch_features, _lengths, batch_labels in loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            logits = self.model(batch_features)
            loss = self.loss_fn(logits, batch_labels)
            probs = torch.sigmoid(logits)

            batch_size_actual = batch_labels.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_examples += batch_size_actual

            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(batch_labels.detach().cpu().int().tolist())

        predicted = [1 if prob >= 0.5 else 0 for prob in all_probs]
        accuracy = float(np.mean(np.asarray(predicted) == np.asarray(all_labels))) if all_labels else 0.0
        eer = compute_eer(all_labels, all_probs)

        return {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": accuracy,
            "eer": eer,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Persist model weights and minimal config."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            output_path,
        )
        return output_path

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "AntiSpoofClassifier":
        """Load a previously saved classifier checkpoint."""
        checkpoint = torch.load(Path(path), map_location=device or "cpu")
        classifier = cls(
            model_type=checkpoint["model_type"],
            learning_rate=checkpoint.get("learning_rate", 1e-3),
            weight_decay=checkpoint.get("weight_decay", 1e-4),
            device=device,
        )
        classifier.model.load_state_dict(checkpoint["state_dict"])
        classifier.model.to(classifier.device)
        return classifier

    def _build_loader(
        self,
        features: Sequence[FeatureInput],
        labels: Sequence[int],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[SpoofSample]:
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")
        samples = [
            SpoofSample(features=self._to_numpy(feature), label=int(label))
            for feature, label in zip(features, labels)
        ]
        dataset = FeatureDataset(samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_feature_batch,
        )

    @staticmethod
    def _to_numpy(features: FeatureInput) -> np.ndarray:
        matrix = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
        if matrix.ndim != 2:
            raise ValueError("Each feature input must be a 2D matrix of shape (frames, coeffs).")
        return matrix.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"AntiSpoofClassifier(model_type={self.model_type!r}, "
            f"device={self.device!r}, learning_rate={self.learning_rate}, "
            f"weight_decay={self.weight_decay})"
        )

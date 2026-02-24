import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.util.gpu import get_device

from .SevenSegmentScorePreprocessor import (
    PreprocessorConfig,
    SevenSegmentScorePreprocessor,
)

NUM_CLASSES = 16


class SevenSegmentReader:
    """
    Inference wrapper for the trained MobileNetV2 seven-segment classifier.

    Supports single image and batch inference. Input images should be raw
    BGR uint8 crops as extracted from the video frame — the preprocessor
    is applied internally so the caller does not need to handle it.

    Usage:
        reader = SevenSegmentReader("checkpoints/best_model.pt")

        # Single image
        score, confidence = reader.read(roi)

        # Batch
        results = reader.read_batch([roi_left, roi_right])
    """

    def __init__(
        self,
        checkpoint_path: str = "src/seven_segment/model.pt",
        preprocessor_config: PreprocessorConfig = None,
        otsu_ratio: float = 0.5,
        device: str = None,
    ):
        self.otsu_ratio = otsu_ratio
        self.preprocessor = SevenSegmentScorePreprocessor(
            preprocessor_config or PreprocessorConfig()
        )
        self.device = torch.device(device or get_device())
        self.model = self._load_model(checkpoint_path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> tuple[int, float]:
        """
        Run inference on a single BGR uint8 crop.
        Returns (predicted_class, confidence) where confidence is the
        softmax probability of the predicted class.
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)  # 1x1xHxW
        scores, classes = self._infer(tensor)
        return int(classes[0]), float(scores[0])

    def read_batch(self, images: list[np.ndarray]) -> list[tuple[int, float]]:
        """
        Run inference on a list of BGR uint8 crops in a single forward pass.
        More efficient than calling read() in a loop for large batches.
        Returns a list of (predicted_class, confidence) tuples in input order.
        """
        if not images:
            return []
        batch = torch.stack([self._preprocess(img) for img in images]).to(
            self.device
        )  # NxCxHxW
        scores, classes = self._infer(batch)
        return [(int(c), float(s)) for c, s in zip(classes, scores)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        model = _build_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Apply preprocessor pipeline and convert to normalised tensor."""
        processed = self.preprocessor.process(image, otsu_ratio=self.otsu_ratio)
        return self.transform(processed)  # 1xHxW float

    @torch.inference_mode()
    def _infer(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass and return (confidences, predicted_classes)."""
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        confidences, predicted = probs.max(dim=1)
        return confidences.cpu().numpy(), predicted.cpu().numpy()


# ---------------------------------------------------------------------------
# Model definition — must match train.py exactly
# ---------------------------------------------------------------------------


def _build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=None)  # weights loaded from checkpoint

    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model

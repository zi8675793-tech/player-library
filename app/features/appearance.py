"""512-dimensional appearance features and quality-weighted track medoids.

``AppearanceExtractor().extract(rgb)`` uses pretrained ResNet-18 pooled features
as a general appearance baseline, not a sports-trained ReID model. Install the
optional backend with ``uv pip install torch torchvision``. Official weights
must already be in the Torch cache unless ``allow_download=True`` is explicit.
For sports ReID, pass a callable ``encoder(rgb_uint8) -> array[512]`` and an
accurate ``model_name``. ``extract_track(images)`` chooses an observed medoid.

The frozen ``AppearanceEmbedding`` fields reproduce docs/data-contracts.md.
They live here because the current app/contracts package has no T1 model and
this task's write scope is limited to the four feature implementation/test files.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from app.features.jersey_number import assess_image_quality, validate_rgb_image

_BASELINE_MODEL = "torchvision-resnet18-imagenet1k-v1"


@dataclass(frozen=True)
class AppearanceEmbedding:
    vector: np.ndarray
    quality_score: float
    model_name: str


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    """Validate a real 512-vector and normalize without float32 overflow."""
    vector = np.asarray(value)
    if vector.shape != (512,) or vector.dtype.kind not in "iuf":
        raise ValueError("Appearance encoder must return a real vector with shape (512,)")
    vector = vector.astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Appearance encoder returned non-finite values")
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError("Appearance encoder returned a zero vector")
    scaled = vector / scale
    result = np.asarray(scaled / np.linalg.norm(scaled), dtype=np.float32)
    result.setflags(write=False)
    return result


class AppearanceExtractor:
    """Lazy pretrained baseline or an explicitly named 512-dimensional encoder.

    ``weights_path`` accepts a trusted ResNet-18 state dict, loaded with
    ``weights_only=True``; provide a custom ``model_name`` for those weights.
    No backend is imported, initialized, or downloaded at construction time.
    Unusable images raise ``ValueError`` rather than fabricating a feature.
    """

    def __init__(
        self,
        encoder: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        model_name: str | None = None,
        allow_download: bool = False,
        device: str = "cpu",
        weights_path: str | Path | None = None,
    ) -> None:
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable")
        if encoder is not None and weights_path is not None:
            raise ValueError("Choose an injected encoder or weights_path")
        if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
            raise ValueError("model_name must be a nonempty, accurate model identifier")
        if (encoder is not None or weights_path is not None) and model_name is None:
            raise ValueError("A custom encoder or weights_path requires model_name")
        if encoder is None and weights_path is None and model_name not in (None, _BASELINE_MODEL):
            raise ValueError(f"The default baseline's model_name is {_BASELINE_MODEL!r}")
        self.model_name = model_name or _BASELINE_MODEL
        self._encoder = encoder
        self._allow_download = allow_download
        self._device = device
        self._weights_path = Path(weights_path) if weights_path is not None else None

    def _load_encoder(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._encoder is not None:
            return self._encoder
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "The default appearance backend needs compatible torch and torchvision; "
                "install them with `uv pip install torch torchvision` or inject an encoder"
            ) from exc

        weights = ResNet18_Weights.IMAGENET1K_V1
        if self._weights_path is not None:
            if not self._weights_path.is_file():
                raise FileNotFoundError(f"Appearance weights not found: {self._weights_path}")
            state_dict = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        else:
            cached = (
                Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
            )
            if cached.is_file():
                state_dict = torch.load(cached, map_location="cpu", weights_only=True)
            elif self._allow_download:
                state_dict = weights.get_state_dict(progress=False, check_hash=True)
            else:
                raise RuntimeError(
                    "Pretrained ResNet-18 weights are not cached. Populate the Torch cache, "
                    "provide weights_path and model_name, or explicitly set allow_download=True"
                )

        model = resnet18(weights=None)
        model.load_state_dict(state_dict, strict=True)
        model.fc = torch.nn.Identity()
        model.eval().to(self._device)
        preprocess = weights.transforms()

        def encode(image: np.ndarray) -> np.ndarray:
            tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                return model(tensor)[0].detach().cpu().numpy()

        self._encoder = encode
        return encode

    def _extract_usable(self, image: np.ndarray, quality: float) -> AppearanceEmbedding:
        vector = _normalize_vector(self._load_encoder()(image))
        return AppearanceEmbedding(vector=vector, quality_score=quality, model_name=self.model_name)

    def extract(self, image: np.ndarray) -> AppearanceEmbedding:
        """Extract a finite, float32, L2-normalized vector from a usable RGB image."""
        image = validate_rgb_image(image)
        quality = assess_image_quality(image)
        if quality <= 0:
            raise ValueError("No usable appearance image: crop is blurred, backlit, or too small")
        return self._extract_usable(image, quality)

    def extract_track(self, images: Iterable[np.ndarray]) -> AppearanceEmbedding:
        """Select a quality-weighted cosine medoid, preserving a real frame vector."""
        embeddings = []
        for image in images:
            image = validate_rgb_image(image)
            quality = assess_image_quality(image)
            if quality > 0:
                embeddings.append(self._extract_usable(image, quality))
        if not embeddings:
            raise ValueError("No usable appearance images in track")

        vectors = np.stack([item.vector for item in embeddings]).astype(np.float64)
        qualities = np.asarray([item.quality_score for item in embeddings], dtype=np.float64)
        # Sum of weighted cosine similarities; avoids allocating an N-by-N matrix.
        similarities = vectors @ (qualities @ vectors)
        best = max(range(len(embeddings)), key=lambda i: (similarities[i], qualities[i], -i))
        return embeddings[best]

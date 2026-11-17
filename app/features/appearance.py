"""512-dimensional appearance features and quality-weighted track medoids.

``AppearanceExtractor().extract(rgb)`` uses pretrained OSNet x1.0 (SportsMOT)
features as the appearance model for sports player tracking and ReID.
Pretrained weights are stored locally at ``models/osnet_x1_0_sportsmot.pth``.
For custom ReID backends, pass a callable ``encoder(rgb_uint8) -> array[512]``
and an accurate ``model_name``. ``extract_track(images)`` chooses an observed medoid.

The frozen ``AppearanceEmbedding`` fields reproduce docs/data-contracts.md.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.features.jersey_number import assess_image_quality, validate_rgb_image

_BASELINE_MODEL = "sports-osnet-x1.0-sportsmot"
_LEGACY_BASELINE_MODEL = "torchvision-resnet18-imagenet1k-v1"


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
    """Lazy pretrained sports OSNet x1.0 baseline or an explicitly named 512-dimensional encoder.

    ``weights_path`` accepts a trusted state dict, loaded with
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
        if (
            encoder is None
            and weights_path is None
            and model_name not in (None, _BASELINE_MODEL, _LEGACY_BASELINE_MODEL)
        ):
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
            import torchvision.transforms as t_transforms

            from app.features.models.osnet import build_osnet
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "The default appearance backend needs compatible torch and torchvision; "
                "install them with `uv pip install torch torchvision` or inject an encoder"
            ) from exc

        weights_path = self._weights_path
        if weights_path is None:
            project_weights = (
                Path(__file__).resolve().parents[2] / "models" / "osnet_x1_0_sportsmot.pth"
            )
            cwd_weights = Path("models/osnet_x1_0_sportsmot.pth").resolve()
            if project_weights.is_file():
                weights_path = project_weights
            elif cwd_weights.is_file():
                weights_path = cwd_weights
            else:
                raise FileNotFoundError(f"OSNet weights not found at {project_weights}")
        elif not weights_path.is_file():
            raise FileNotFoundError(f"Appearance weights not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if any("conv2a" in k for k in state_dict):
            model = build_osnet("x1_0", num_classes=0)
            model.load_state_dict(state_dict, strict=True)
            model.eval().to(self._device)
            is_osnet = True
            backbone = None
        else:
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            model.load_state_dict(state_dict, strict=True)
            backbone = torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            backbone.eval().to(self._device)
            is_osnet = False

        preprocess = t_transforms.Compose(
            [
                t_transforms.Resize((256, 128)),
                t_transforms.ToTensor(),
                t_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        w_horiz = (
            torch.tensor([1.0, 1.5, 1.5, 1.0], device=self._device).view(1, 1, 4)
            if not is_osnet
            else None
        )

        def encode(image: np.ndarray) -> np.ndarray:
            tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                if is_osnet:
                    feat = model(tensor)[0]
                    normed = torch.nn.functional.normalize(feat, dim=0)
                    return normed.detach().cpu().numpy()
                else:
                    fmap = backbone(tensor)[0]
                    fmap_w = fmap * w_horiz
                    p_glob = torch.nn.functional.normalize(fmap_w.mean(dim=(1, 2)), dim=0)
                    p_top = torch.nn.functional.normalize(fmap_w[:, 0:3, :].mean(dim=(1, 2)), dim=0)
                    p_mid = torch.nn.functional.normalize(fmap_w[:, 3:5, :].mean(dim=(1, 2)), dim=0)
                    p_bot = torch.nn.functional.normalize(fmap_w[:, 5:8, :].mean(dim=(1, 2)), dim=0)

                    fused = 0.10 * p_glob + 0.50 * p_top + 0.10 * p_mid + 0.30 * p_bot
                    fused = torch.nn.functional.normalize(fused, dim=0)
                    return fused.detach().cpu().numpy()

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
        effective_weights = []
        for image in images:
            image = validate_rgb_image(image)
            quality = assess_image_quality(image)
            if quality > 0:
                h, w = image.shape[:2]
                aspect = h / max(w, 1)
                aspect_weight = float(np.exp(-((aspect - 2.0) ** 2) / (2 * (1.0**2))))
                area = h * w
                scale_weight = float((area / 25000.0) ** 1.5)
                eff_weight = (quality**1.5) * aspect_weight * scale_weight
                embeddings.append(self._extract_usable(image, quality))
                effective_weights.append(max(eff_weight, 1e-4))
        if not embeddings:
            raise ValueError("No usable appearance images in track")

        vectors = np.stack([item.vector for item in embeddings]).astype(np.float64)
        weights = np.asarray(effective_weights, dtype=np.float64)
        # Sum of weighted cosine similarities; avoids allocating an N-by-N matrix.
        similarities = vectors @ (weights @ vectors)
        best = max(
            range(len(embeddings)),
            key=lambda i: (similarities[i], embeddings[i].quality_score, -i),
        )
        return embeddings[best]

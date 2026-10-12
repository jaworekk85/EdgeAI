"""Shared CPU image preparation; no Torch dependency."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from edge_ai.utils.serialization import positive, resolution, validate_types


@dataclass(frozen=True)
class PreprocessingSpec:
    resize_short_edge: int = 256
    crop: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: Literal["bilinear"] = "bilinear"
    layout: Literal["NCHW"] = "NCHW"
    color: Literal["RGB"] = "RGB"

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.resize_short_edge, "resize_short_edge")
        resolution(self.crop)
        if self.resize_short_edge < max(self.crop):
            raise ValueError("resize must cover the crop")
        for value in self.std:
            positive(value, "normalization std")


def preprocess_image(path: Path, spec: PreprocessingSpec):
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width <= height:
            size = (spec.resize_short_edge, int(height * spec.resize_short_edge / width))
        else:
            size = (int(width * spec.resize_short_edge / height), spec.resize_short_edge)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        crop_h, crop_w = spec.crop
        left, top = round((size[0] - crop_w) / 2), round((size[1] - crop_h) / 2)
        image = image.crop((left, top, left + crop_w, top + crop_h))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    array = (array - np.asarray(spec.mean, dtype=np.float32)[:, None, None]) / np.asarray(
        spec.std,
        dtype=np.float32,
    )[:, None, None]
    return np.ascontiguousarray(array[None], dtype=np.float32)

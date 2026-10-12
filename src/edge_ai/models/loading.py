"""Explicit, offline model construction with lazily imported Torchvision."""

from pathlib import Path

from edge_ai.models.variant import ModelVariant


def load_pytorch_model(variant: ModelVariant, checkpoint: Path):
    import torch
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

    if variant.family != "mobilenet_v3" or variant.precision != "fp32":
        raise ValueError("CPU reference supports MobileNetV3 FP32")
    if variant.architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
    elif variant.architecture == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
    else:
        raise ValueError("unsupported model architecture")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.cpu().eval()

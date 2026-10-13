"""Explicit, offline model construction with lazily imported Torchvision."""

from pathlib import Path

from edge_ai.models.variant import ModelVariant


def load_pytorch_model(variant: ModelVariant, checkpoint: Path):
    import torch
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

    if variant.family != "mobilenet_v3" or variant.precision not in ("fp32", "fp16"):
        raise ValueError("PyTorch CPU reference supports MobileNetV3 FP32/FP16")
    if variant.architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
    elif variant.architecture == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=None)
    else:
        raise ValueError("unsupported model architecture")
    if variant.precision == "fp16":
        model = model.half()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = torch.float16 if variant.precision == "fp16" else torch.float32
    if any(value.is_floating_point() and value.dtype != expected for value in state.values()):
        raise ValueError("checkpoint floating-point dtype differs from declared precision")
    model.load_state_dict(state)
    return model.cpu().eval()

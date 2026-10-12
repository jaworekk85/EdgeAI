"""Shared partial nominal counter: Conv2d/Linear only, 1 MAC = 2 FLOPs."""

from dataclasses import replace

from edge_ai.compression.spec import CompressionSpec


def count_nominal(model, input_shape: tuple[int, ...]) -> CompressionSpec:
    import torch

    flops = 0
    handles = []

    def count(module, inputs, output):
        nonlocal flops
        if isinstance(module, torch.nn.Conv2d):
            operations = 2 * (module.in_channels // module.groups)
            operations *= module.kernel_size[0] * module.kernel_size[1]
        else:
            operations = 2 * module.in_features
        if module.bias is not None:
            operations += 1
        flops += output.numel() * operations

    try:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(count))
        with torch.inference_mode():
            model(torch.zeros(input_shape, dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    return replace(
        CompressionSpec(),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        flops=float(flops),
        counting_input_shape=input_shape,
        counting_method="partial_conv2d_linear; MAC=2 FLOPs; bias_add=1 FLOP",
        counting_exclusions=("normalization", "activations", "pooling", "elementwise", "other_ops"),
    )

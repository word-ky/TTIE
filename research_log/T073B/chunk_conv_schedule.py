"""Prospective row scheduling for PromptIR's 4K depthwise convolution.

Only the spatial execution schedule changes; the same Conv2d module, weights,
padding, and output positions are used for each chunk. This is experimental
until synthetic equivalence and full-resolution execution are demonstrated.
"""

import torch
from torch import nn


def schedule_depthwise(model, rows=256, threshold=250_000_000):
    scheduled = []
    for name, module in model.named_modules():
        if not (name.endswith("qkv_dwconv") or name.endswith("ffn.dwconv")):
            continue
        assert isinstance(module, nn.Conv2d)
        assert module.kernel_size == (3, 3)
        assert module.stride == (1, 1)
        assert module.padding == (1, 1)
        assert module.dilation == (1, 1)
        assert module.groups == module.in_channels == module.out_channels
        original_forward = module.forward

        def row_forward(x, original=original_forward, output_channels=module.out_channels):
            height = x.shape[-2]
            if x.numel() + x.shape[0] * output_channels * height * x.shape[-1] < threshold:
                return original(x)
            parts = []
            for start in range(0, height, rows):
                stop = min(height, start + rows)
                source_start = max(0, start - 1)
                source_stop = min(height, stop + 1)
                chunk = original(x[:, :, source_start:source_stop, :].contiguous())
                parts.append(chunk[:, :, start - source_start:start - source_start + stop - start, :])
            return torch.cat(parts, dim=2)

        module.forward = row_forward
        scheduled.append(name)
    return scheduled


def schedule_feedforward(model, rows=64, threshold=250_000_000):
    scheduled = []
    for name, module in model.named_modules():
        if not name.endswith("ffn"):
            continue
        original_forward = module.forward
        assert module.dwconv.kernel_size == (3, 3)
        assert module.dwconv.padding == (1, 1)
        assert module.dwconv.stride == (1, 1)

        def row_forward(x, ffn=module, original=original_forward):
            if x.numel() < threshold:
                return original(x)
            height = x.shape[-2]
            parts = []
            for start in range(0, height, rows):
                stop = min(height, start + rows)
                source_start = max(0, start - 1)
                source_stop = min(height, stop + 1)
                window = x[:, :, source_start:source_stop, :].contiguous()
                projected = ffn.project_in(window)
                first, second = ffn.dwconv(projected).chunk(2, dim=1)
                filtered = torch.nn.functional.gelu(first) * second
                result = ffn.project_out(filtered)
                parts.append(result[:, :, start - source_start:start - source_start + stop - start, :])
            return torch.cat(parts, dim=2)

        module.forward = row_forward
        scheduled.append(name)
    return scheduled

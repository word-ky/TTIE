"""Execution-only query-row schedule for the frozen SNR-Aware attention."""
import torch
import torch.nn.functional as F

QUERY_ROWS = 512


def forward(self, q, k, v, mask=None):
    keys = k.transpose(2, 3)
    output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=v.dtype, device=v.device)
    for first in range(0, q.shape[-2], QUERY_ROWS):
        last = min(first + QUERY_ROWS, q.shape[-2])
        attn = torch.matmul(q[:, :, first:last, :] / self.temperature, keys)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output[:, :, first:last, :] = torch.matmul(attn, v)
    # All repository call sites discard the returned attention matrix.
    return output, None


def install():
    from models.archs.transformer.Modules import ScaledDotProductAttention
    ScaledDotProductAttention.forward = forward

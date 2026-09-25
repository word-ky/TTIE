"""Low-only-free, deterministic equivalence tests against frozen official attention."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from chunk_attention import forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-modules", type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_snr_attention", a.official_modules)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attention = module.ScaledDotProductAttention(temperature=8.0, attn_dropout=0.0).eval()
    torch.manual_seed(702)
    results = []
    for length in (31, 513, 1025):
        q = torch.randn(1, 2, length, 64, dtype=torch.float32)
        k = torch.randn(1, 2, length, 64, dtype=torch.float32)
        v = torch.randn(1, 2, length, 64, dtype=torch.float32)
        for label, mask in (("none", None),
                            ("mixed", (torch.arange(length) % 3 != 0).float().reshape(1, 1, 1, length)),
                            ("all_zero", torch.zeros(1, 1, 1, length))):
            original, _ = attention(q, k, v, mask)
            proposed, omitted = forward(attention, q, k, v, mask)
            delta = (original - proposed).abs().max().item()
            assert omitted is None and torch.allclose(original, proposed, atol=1e-6, rtol=1e-5), (length, label, delta)
            results.append({"tokens": length, "mask": label, "max_abs_delta": delta})
    print(json.dumps({"classification": "PASS_QUERY_ROW_EQUIVALENCE", "cases": results}))


if __name__ == "__main__":
    main()

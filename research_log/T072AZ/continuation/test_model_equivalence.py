"""Synthetic full-checkpoint equivalence; never opens UHD-LL inputs or references."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "runtime"))
bindings = json.loads((HERE.parent / "seal/baseline_bindings.json").read_text())
row = bindings["snr_aware"]
from ttie.snr_exporter import load_model
from chunk_attention import install


def main():
    torch.manual_seed(702)
    model = load_model(row["checkpoint"], str(ROOT / "runtime" / row["config"]), row["source"])
    image = torch.rand(1, 3, 64, 64, device="cuda", dtype=torch.float32)
    mask = torch.rand(1, 1, 64, 64, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        reference = model(image, mask)
        install()
        proposed = model(image, mask)
    delta = (reference - proposed).abs().max().item()
    assert torch.allclose(reference, proposed, atol=1e-6, rtol=1e-5), delta
    print(json.dumps({"classification": "PASS_FULL_MODEL_SYNTHETIC_EQUIVALENCE",
                      "shape": list(reference.shape), "max_abs_delta": delta,
                      "checkpoint_sha256": "432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781",
                      "reference_reads": 0, "metrics": 0}))


if __name__ == "__main__":
    main()

"""Unchanged frozen SNR exporter entrypoint with prospective attention schedule."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / "runtime"
SEAL = HERE.parent / "seal"
sys.path.insert(0, str(ROOT))
bindings = json.loads((SEAL / "baseline_bindings.json").read_text())
sys.path.insert(0, bindings["snr_aware"]["source"])
from chunk_attention import install

install()
from ttie.snr_exporter import main

if __name__ == "__main__":
    main()

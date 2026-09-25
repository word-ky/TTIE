"""Reuse the sealed T072-AZ low-only smoke worker with query-row attention."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = HERE.parent / "seal"
sys.path.insert(0, str(SEAL))
bindings = json.loads((SEAL / "baseline_bindings.json").read_text())
sys.path.insert(0, bindings["snr_aware"]["source"])
from chunk_attention import install

install()
from worker import main

if __name__ == "__main__":
    main()

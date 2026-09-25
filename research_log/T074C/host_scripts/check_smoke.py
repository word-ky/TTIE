"""Execution-only smoke check (geometry, dtype, finiteness, hashes); no quality inspection."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/autodl-tmp/TTIE/T074B/code")
from verify_generic_outputs import verify_row, sha256_file
receipt, out = Path(sys.argv[1]), Path(sys.argv[2])
files = json.loads(receipt.read_bytes())["files"]
m = json.loads((out / "output_manifest.json").read_bytes())
assert m["reference_reads"] == 0 and m["metrics"] == 0
rows = m["rows"]
assert 1 <= len(rows) < len(files)
by = {f["name"]: f for f in files}
res = []
for row in rows:
    size = verify_row(row, by[row["low_name"]], out)
    res.append({k: row.get(k) for k in ("low_name", "shape", "dtype", "whole_run_seconds", "peak_gpu_memory_bytes", "decision_status")})
print(json.dumps({"smoke_ok": True, "promotable": m.get("promotable"), "count": len(rows),
                  "manifest_sha256": sha256_file(out / "output_manifest.json"), "rows": res}, indent=1))

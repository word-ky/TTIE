"""Independent post-run freeze of 150+150 native float outputs; no references."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/TTIE/T072AZ")
RUNS = Path("/root/autodl-fs/TTIE/T072AZ/runs")
EV = ROOT / "T072AZ"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    status = json.loads((RUNS / "status.json").read_text())
    assert status["classification"] == "FULL_PROCESSES_DONE_AWAITING_INDEPENDENT_VERIFY"
    assert status["smoke"]["classification"] == "UHDLL_NATIVE_BASELINES_SMOKE_PASS"
    expected = {r["name"]: r["sha256"] for r in json.loads((EV / "low_receipt.json").read_text())["files"]}
    binding = json.loads((EV / "seal/baseline_bindings.json").read_text())
    manifest = {"classification": "UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090",
                "reference_reads": 0, "metrics": 0, "methods": {}}
    for method in ("retinexformer", "snr_aware"):
        directory = RUNS / "full" / method
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["mode"] == binding[method]["mode"]
        if method == "retinexformer":
            assert receipt["GT_mean"] is False and receipt["self_ensemble"] is False
        else:
            assert receipt["parameter_hash_before"] == receipt["parameter_hash_after"] == binding[method]["accepted_parameter_hash"]
        rows = receipt["rows"]
        assert len(rows) == 150
        assert {Path(r["low"]).name for r in rows} == set(expected)
        assert {p.name for p in directory.glob("*.npy")} == {Path(n).stem + ".npy" for n in expected}
        frozen = []
        for row in rows:
            name = Path(row["low"]).name
            assert row["low_sha256"] == expected[name]
            assert Path(row["low"]) == ROOT / "input/T072I_input_cache" / name
            file = directory / (Path(name).stem + ".npy")
            y = np.load(file, allow_pickle=False)
            assert y.shape == (2160, 3840, 3) and y.dtype == np.float32
            assert bool(np.isfinite(y).all()) and row["finite"] is True
            assert row["shape"] == [2160, 3840, 3]
            pixel_hash = hashlib.sha256(y.tobytes()).hexdigest()
            assert pixel_hash == row["output_sha256"]
            assert row["runtime_s"] >= 0 and row["peak_memory_bytes"] > 0
            frozen.append({"name": name, "low_sha256": expected[name],
                           "output_sha256": pixel_hash, "file_sha256": sha(file),
                           "shape": row["shape"], "dtype": str(y.dtype), "finite": True,
                           "runtime_s": row["runtime_s"], "peak_memory_bytes": row["peak_memory_bytes"]})
        manifest["methods"][method] = {"count": 150, "receipt_sha256": sha(directory / "receipt.json"),
                                       "rows": frozen, "total_runtime_s": sum(r["runtime_s"] for r in frozen),
                                       "max_peak_memory_bytes": max(r["peak_memory_bytes"] for r in frozen)}
    path = RUNS / "output_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"classification": manifest["classification"], "counts": {m: x["count"] for m, x in manifest["methods"].items()},
                      "manifest_sha256": sha(path), "reference_reads": 0, "metrics": 0}))


if __name__ == "__main__":
    main()

"""Independent full-output audit for prospectively sealed T072-AZ continuation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/TTIE/T072AZ")
EV = ROOT / "T072AZ"
RUNS = Path("/root/autodl-fs/TTIE/T072AZ/runs")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    status = json.loads((RUNS / "continuation_status.json").read_text())
    assert status["classification"] == "FULL_PROCESSES_DONE_AWAITING_INDEPENDENT_VERIFY"
    assert status["snr_smoke_count"] == 1 and status["snr_smoke"]["shape"] == [2160, 3840, 3]
    assert len(status["full"]) == 2 and all(x["returncode"] == 0 for x in status["full"])
    old = json.loads((RUNS / "smoke/receipt.json").read_text())
    assert old["runs"][0]["method"] == "retinexformer" and old["runs"][0]["status"] == "FROZEN"
    bindings = json.loads((EV / "seal/baseline_bindings.json").read_text())
    expected = {row["name"]: row["sha256"] for row in json.loads((EV / "low_receipt.json").read_text())["files"]}
    manifest = {"classification": "UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090",
                "reference_reads": 0, "metrics": 0, "methods": {}}
    for method in ("retinexformer", "snr_aware"):
        output = RUNS / "full" / method
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["mode"] == bindings[method]["mode"]
        if method == "retinexformer":
            assert receipt["GT_mean"] is False and receipt["self_ensemble"] is False
        else:
            assert receipt["parameter_hash_before"] == receipt["parameter_hash_after"] == bindings[method]["accepted_parameter_hash"]
        rows = receipt["rows"]
        assert len(rows) == 150 and {Path(row["low"]).name for row in rows} == set(expected)
        assert {p.name for p in output.glob("*.npy")} == {Path(n).stem + ".npy" for n in expected}
        frozen = []
        for row in rows:
            name = Path(row["low"]).name
            assert Path(row["low"]) == ROOT / "input/T072I_input_cache" / name
            assert row["low_sha256"] == expected[name]
            path = output / (Path(name).stem + ".npy")
            array = np.load(path, allow_pickle=False)
            assert array.shape == (2160, 3840, 3) and array.dtype == np.float32
            assert np.isfinite(array).all() and row["finite"] is True
            assert row["shape"] == [2160, 3840, 3]
            output_sha = hashlib.sha256(array.tobytes()).hexdigest()
            assert output_sha == row["output_sha256"]
            assert row["runtime_s"] >= 0 and row["peak_memory_bytes"] > 0
            frozen.append({"name": name, "low_sha256": expected[name], "output_sha256": output_sha,
                           "file_sha256": sha(path), "shape": row["shape"], "dtype": str(array.dtype),
                           "finite": True, "runtime_s": row["runtime_s"],
                           "peak_memory_bytes": row["peak_memory_bytes"]})
        manifest["methods"][method] = {"count": 150, "receipt_sha256": sha(output / "receipt.json"),
                                       "rows": frozen, "total_runtime_s": sum(r["runtime_s"] for r in frozen),
                                       "max_peak_memory_bytes": max(r["peak_memory_bytes"] for r in frozen)}
    path = RUNS / "continuation_output_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"classification": manifest["classification"],
                      "manifest_sha256": sha(path), "counts": {m: 150 for m in manifest["methods"]},
                      "reference_reads": 0, "metrics": 0}))


if __name__ == "__main__":
    main()

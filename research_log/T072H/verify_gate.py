from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify() -> dict:
    receipt = json.loads((ROOT / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["task"] == "T072-H"
    assert receipt["status"] == "BLOCKED"
    assert receipt["classification"] == "BLOCKED"
    assert receipt["smoke_input"]["decoded"] is False
    gate = receipt["gpu_gate"]
    minimum = gate["requirements"]["minimum_free_mib"]
    maximum_external = gate["requirements"]["maximum_unrelated_process_mib"]
    assert gate["qualifying_device"] is None
    for device in gate["devices"]:
        assert device["free_mib"] < minimum
        assert any(p["used_mib"] > maximum_external for p in device["unrelated_processes"])
        assert device["qualifies"] is False
    accounting = receipt["accounting"]
    for key in ("inference_runs", "optimizer_runs", "model_fits", "reference_reads", "metrics", "smoke_input_decodes"):
        assert accounting[key] == 0
    assert accounting["retinexformer_status"] == "UNRUN"
    assert accounting["snr_aware_status"] == "UNRUN"
    return {
        "status": "verified",
        "classification": receipt["classification"],
        "free_mib": [d["free_mib"] for d in gate["devices"]],
        "reference_reads": accounting["reference_reads"],
        "metrics": accounting["metrics"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))

"""Independent verifier for the T072-J pre-launch BLOCKED receipt."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "receipt.json"

EXPECTED_BINDINGS = {
    "retinexformer": {
        "upstream_commit": "1e9a0efce4b306b6701b824768370ff26066c32a",
        "checkpoint_sha256": "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b",
        "config_sha256": "5260d0c65878f6a39712f70948be1936d8583531491d832cb59362fffba894ac",
        "accepted_binding_sha256": "a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00",
        "mode": "default_no_gt_mean",
        "options": {"GT_mean": False, "self_ensemble": False},
    },
    "snr_aware": {
        "upstream_commit": "1113144c82adc8bcc4a9ec27749ed75f196a4e4d",
        "checkpoint_sha256": "432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781",
        "config_sha256": "fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb",
        "parameter_sha256": "11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4",
        "accepted_binding_sha256": "03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875",
        "mode": "ttie_native_pad16",
    },
}

EXPECTED_SMOKE = {
    "name": "1003_UHD_LL.JPG",
    "sha256": "cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca",
    "bytes": 752975,
    "geometry_wh": [3840, 2160],
    "mode": "RGB",
}


def verify() -> None:
    data = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert data["task"] == "T072-J"
    assert data["status"] == "DONE"
    assert data["classification"] == "BLOCKED"
    assert data["bindings"] == EXPECTED_BINDINGS
    smoke = data["smoke_input"]
    for key, value in EXPECTED_SMOKE.items():
        assert smoke[key] == value, (key, smoke[key], value)
    assert smoke["source_task"] == "T072-I"
    assert smoke["payload_opened_t072j"] is False
    assert smoke["decoded_t072j"] is False

    gate = data["gate"]
    assert gate["qualifying_devices"] == []
    assert gate["processes_disturbed"] is False
    assert gate["requirement"]["minimum_free_mib"] == 40960
    assert gate["requirement"]["maximum_unrelated_process_mib"] == 1024
    assert any("2078 MiB" in row for row in gate["query_gpu_csv"])
    assert any("3499 MiB" in row for row in gate["query_gpu_csv"])
    assert sum("VLLM::Worker" in row and "44972 MiB" in row for row in gate["query_compute_apps_csv"]) == 2

    accounting = data["accounting"]
    assert accounting["retinexformer"] == {"status": "UNRUN", "allowed_runs": 1, "run_count": 0}
    assert accounting["snr_aware"] == {"status": "UNRUN", "allowed_runs": 1, "run_count": 0}
    assert accounting["final_ours"] == {"status": "NOT_RUN_T072J", "run_count": 0}
    for key in (
        "inference_runs",
        "model_launches",
        "optimizer_runs",
        "model_fits",
        "reference_reads",
        "clean_target_reads",
        "metrics",
        "input_payload_reads",
        "input_decodes",
    ):
        assert accounting[key] == 0, (key, accounting[key])
    assert data["outputs"] == []
    assert all(value == 0 for value in data["read_ledger"].values())
    assert data["tests"]["model_tests"] == "not_run_by_gate"
    print("T072J_BLOCKED_RECEIPT_PASS")


if __name__ == "__main__":
    verify()

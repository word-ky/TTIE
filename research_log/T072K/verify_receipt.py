"""Independent verifier for a T072-K BLOCKED availability-watch receipt."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECEIPT = ROOT / "receipt.json"
WATCH = ROOT / "watch.jsonl"

EXPECTED_SMOKE = {
    "source_task": "T072-I",
    "name": "1003_UHD_LL.JPG",
    "sha256": "cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca",
    "bytes": 752975,
    "geometry_wh": [3840, 2160],
    "mode": "RGB",
    "payload_opened_t072k": False,
    "decoded_t072k": False,
}
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


def verify(receipt_path: Path = RECEIPT, watch_path: Path = WATCH) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw = watch_path.read_bytes()
    snapshots = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert receipt["task"] == "T072-K"
    assert receipt["status"] == "DONE"
    assert receipt["classification"] == "BLOCKED"
    assert receipt["bindings"] == EXPECTED_BINDINGS
    for key, value in EXPECTED_SMOKE.items():
        assert receipt["smoke_input"][key] == value, (key, receipt["smoke_input"].get(key), value)

    watch = receipt["watch"]
    assert watch["interval_seconds"] == 300
    assert watch["maximum_window_minutes"] == 55
    assert watch["snapshot_count"] == 12
    assert watch["nominal_window_seconds"] == 3300
    assert 3300 <= watch["observed_duration_seconds"] <= 3305
    assert 0 <= watch["clock_overrun_seconds"] <= 5
    assert watch["first_qualifying_snapshot"] is None
    assert watch["event"] == "WINDOW_EXPIRED"
    assert hashlib.sha256(raw).hexdigest() == watch["watch_jsonl_sha256"]
    assert len(snapshots) == 12
    assert [row["seq"] for row in snapshots] == list(range(12))
    assert snapshots[-1]["event"] == "WINDOW_EXPIRED"
    assert all(row["qualifying_indices"] == [] for row in snapshots)
    times = [datetime.fromisoformat(row["utc"].replace("Z", "+00:00")) for row in snapshots]
    assert all((b - a).total_seconds() >= 300 for a, b in zip(times, times[1:]))
    for row in snapshots:
        assert all(gpu["name"] == "NVIDIA RTX A6000" for gpu in row["gpus"])
        assert all(gpu["memory_free_mib"] < 40960 for gpu in row["gpus"])
        assert any(
            app["process_name"].startswith("VLLM::Worker") and app["used_memory_mib"] > 1024
            for app in row["compute_apps"]
        )

    accounting = receipt["accounting"]
    assert accounting["retinexformer"] == {"status": "UNRUN", "allowed_runs": 1, "run_count": 0}
    assert accounting["snr_aware"] == {"status": "UNRUN", "allowed_runs": 1, "run_count": 0}
    assert accounting["final_ours"] == {"status": "NOT_RUN_T072K", "run_count": 0}
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
    assert receipt["outputs"] == []
    assert all(value == 0 for value in receipt["read_ledger"].values())
    print("T072K_BLOCKED_WATCH_RECEIPT_PASS")


if __name__ == "__main__":
    verify()

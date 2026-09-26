"""Build the T072-K blocked receipt from the immutable remote watch log."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WATCH = ROOT / "watch.jsonl"
RECEIPT = ROOT / "receipt.json"

SMOKE = {
    "source_task": "T072-I",
    "source_manifest": "research_log/T072I/dispatch_manifest.json",
    "identity_source": "sealed T072-I manifest row; not reopened or decoded in T072-K",
    "name": "1003_UHD_LL.JPG",
    "sha256": "cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca",
    "bytes": 752975,
    "geometry_wh": [3840, 2160],
    "mode": "RGB",
    "remote_path": "/media/wenchang/F/wjq/TTIE/shared/t072i/uhdll/input/1003_UHD_LL.JPG",
    "payload_opened_t072k": False,
    "decoded_t072k": False,
}

BINDINGS = {
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


def main() -> None:
    raw = WATCH.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    separate_terminal = records[-1].get("kind") == "window_event"
    snapshots = records[:-1] if separate_terminal else records
    terminal = records[-1]
    assert len(snapshots) in (11, 12)
    assert terminal["event"] == "WINDOW_EXPIRED"
    assert all(not row["qualifying_indices"] for row in snapshots)
    first = datetime.fromisoformat(snapshots[0]["utc"].replace("Z", "+00:00"))
    last = datetime.fromisoformat(terminal["utc"].replace("Z", "+00:00"))
    observed_seconds = (last - first).total_seconds()
    receipt = {
        "task": "T072-K",
        "status": "DONE",
        "classification": "BLOCKED",
        "purpose": "Bounded clean-GPU availability watch plus native-4K baseline smoke",
        "authorization_main": "41ff2c429ad2b142ed388fe57045ad9b76b0947f",
        "branch": "codex/T072K-clean-gpu-watch",
        "remote_host": "202.101.162.22",
        "remote_project": "/home/wenchang/asdasdsad/wjq/TTIE",
        "watch": {
            "remote_watch_path": "/media/wenchang/F/wjq/TTIE/shared/t072k/watch/watch.jsonl",
            "local_watch_file": "research_log/T072K/watch.jsonl",
            "interval_seconds": 300,
            "maximum_window_minutes": 55,
            "snapshot_count": len(snapshots),
            "nominal_window_seconds": 3300,
            "observed_duration_seconds": observed_seconds,
            "clock_overrun_seconds": max(0.0, observed_seconds - 3300.0),
            "timing_note": (
                "The terminal event was written at the monotonic deadline without a post-deadline query; "
                "all snapshot gaps remained at least 300 seconds."
                if separate_terminal
                else "The final closure query followed the nominal 55-minute deadline by scheduler/query overhead; all snapshot gaps remained at least 300 seconds."
            ),
            "first_snapshot_utc": snapshots[0]["utc"],
            "last_snapshot_utc": snapshots[-1]["utc"],
            "terminal_event_utc": terminal["utc"],
            "first_qualifying_snapshot": None,
            "event": terminal["event"],
            "watch_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
            "snapshots": snapshots,
            "records": records,
        },
        "smoke_input": SMOKE,
        "bindings": BINDINGS,
        "accounting": {
            "retinexformer": {"status": "UNRUN", "allowed_runs": 1, "run_count": 0},
            "snr_aware": {"status": "UNRUN", "allowed_runs": 1, "run_count": 0},
            "final_ours": {"status": "NOT_RUN_T072K", "run_count": 0},
            "inference_runs": 0,
            "model_launches": 0,
            "optimizer_runs": 0,
            "model_fits": 0,
            "reference_reads": 0,
            "clean_target_reads": 0,
            "metrics": 0,
            "input_payload_reads": 0,
            "input_decodes": 0,
        },
        "outputs": [],
        "read_ledger": {
            "low_payload": 0,
            "gt_reference": 0,
            "clean_reference": 0,
            "target_normalization": 0,
            "metrics": 0,
        },
        "tests": {
            "independent_verifier": "pending",
            "watch_parser": f"{len(snapshots)} snapshots; WINDOW_EXPIRED",
            "model_tests": "not_run_by_gate",
            "remote_input_access": "not performed",
        },
        "next_step": "Await a future qualifying clean-A6000 window; no retry or workaround in T072-K.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshots": len(snapshots), "watch_jsonl_sha256": receipt["watch"]["watch_jsonl_sha256"]}))


if __name__ == "__main__":
    main()

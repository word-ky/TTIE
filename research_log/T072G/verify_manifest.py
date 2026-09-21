from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def canonical_without_root_hash(manifest: dict) -> bytes:
    value = copy.deepcopy(manifest)
    value.pop("root_manifest_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_root = hashlib.sha256(canonical_without_root_hash(manifest)).hexdigest()
    assert manifest["root_manifest_sha256"] == expected_root
    assert manifest["task"] == "T072-G"
    assert manifest["dataset"] == "SMID"
    assert manifest["status"] == "BLOCKED"
    assert manifest["classification"] == "BLOCKED"

    evidence = manifest["evidence_policy"]
    assert evidence["reference_payload_opened"] is False
    assert evidence["reference_reads"] == 0
    assert evidence["metrics"] == 0
    assert evidence["inference_runs"] == 0
    assert evidence["optimizer_runs"] == 0
    assert evidence["low_payload_files_opened"] == 0

    test_list = manifest["test_list"]
    scenes = test_list["scene_ids"]
    assert len(scenes) == test_list["line_count"] == 49
    assert len(set(scenes)) == len(scenes)
    assert all(re.fullmatch(r"\d{4}", scene) for scene in scenes)
    assert test_list["sha256"] == "cd408769f80576bba9d4057add2d7c99e70bbf62682af4d7207885ab0d631fc3"
    assert test_list["byte_count"] == 245

    candidate = manifest["candidate_low_only_manifest"]
    assert candidate["scene_ids_ordered"] == scenes
    assert candidate["low_payload_enumerated"] is False
    assert candidate["low_sha256"] is None
    assert candidate["reference_payload_opened"] is False

    snr = manifest["observed_baselines"]["snr_aware"]
    retinex = manifest["observed_baselines"]["retinexformer"]
    assert snr["low_frame_cap"] == "first 30 paths per scene"
    assert retinex["low_frame_cap"] == "none; all low paths are indexed"
    assert snr["test_list_path_in_code"] != retinex["test_list_path_in_code"]
    assert any("first 30" in blocker for blocker in manifest["blockers"])
    assert any("text_list.txt" in blocker for blocker in manifest["blockers"])
    assert any("ISP" in blocker for blocker in manifest["blockers"])
    assert any("SMID_LQ_np" in blocker for blocker in manifest["blockers"])

    return {
        "status": "verified",
        "classification": manifest["classification"],
        "scene_count": len(scenes),
        "root_manifest_sha256": expected_root,
        "reference_reads": evidence["reference_reads"],
        "metrics": evidence["metrics"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))

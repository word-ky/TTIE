from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from research_log.T072I.verify_dispatch import VerificationError, verify


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
METADATA = HERE / "input_metadata.json"
MANIFEST_PATH = HERE / "dispatch_manifest.json"
INPUT_ROOT = Path(os.environ.get("T072I_INPUT_ROOT", str(WORKTREE.parents[1] / "T072I_input_cache")))


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not INPUT_ROOT.is_dir():
        pytest.fail(f"T072I_INPUT_ROOT is missing: {INPUT_ROOT}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def assert_rejected(candidate: dict, manifest: dict) -> None:
    with pytest.raises(VerificationError):
        verify(candidate, METADATA, INPUT_ROOT)


def test_complete_dispatch_verifies(manifest: dict) -> None:
    result = verify(manifest, METADATA, INPUT_ROOT)
    assert result["classification"] == "UHDLL_FULL_LOW_ONLY_DISPATCH_SEALED"
    assert result["lows"] == 150
    assert result["jobs"] == 450


def test_rejects_missing_job(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["jobs"].pop()
    assert_rejected(candidate, manifest)


def test_rejects_extra_job(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    extra = copy.deepcopy(candidate["jobs"][0])
    extra["job_id"] = "ours:extra.JPG"
    extra["input_name"] = "extra.JPG"
    extra["input_relative_path"] = "extra.JPG"
    extra["output_relative_path"] = "outputs/ours/extra.pt"
    candidate["jobs"].append(extra)
    assert_rejected(candidate, manifest)


def test_rejects_duplicate_input(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["lows"][1]["name"] = candidate["lows"][0]["name"]
    assert_rejected(candidate, manifest)


def test_rejects_altered_binding(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["frozen_bindings"]["ours"]["manifest_sha256"] = "changed"
    assert_rejected(candidate, manifest)


def test_rejects_altered_low_hash(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["lows"][0]["sha256"] = "0" * 64
    assert_rejected(candidate, manifest)


def test_rejects_output_collision(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["jobs"][1]["output_relative_path"] = candidate["jobs"][0]["output_relative_path"]
    assert_rejected(candidate, manifest)


def test_rejects_reference_path(manifest: dict) -> None:
    candidate = copy.deepcopy(manifest)
    candidate["jobs"][0]["input_relative_path"] = "reference/1003_UHD_LL.JPG"
    assert_rejected(candidate, manifest)

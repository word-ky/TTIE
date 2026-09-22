from __future__ import annotations

import json
from pathlib import Path

import pytest

from verify_receipt import verify


HERE = Path(__file__).resolve().parent


def test_receipt_verifier_passes() -> None:
    verify()


def test_tampered_binding_is_rejected(tmp_path: Path) -> None:
    altered = json.loads((HERE / "receipt.json").read_text(encoding="utf-8"))
    altered["bindings"]["retinexformer"]["mode"] = "changed"
    candidate = tmp_path / "receipt.json"
    candidate.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(AssertionError):
        verify(candidate, HERE / "watch.jsonl")


def test_tampered_watch_is_rejected(tmp_path: Path) -> None:
    lines = (HERE / "watch.jsonl").read_text(encoding="utf-8").splitlines()
    altered = json.loads(lines[0])
    altered["qualifying_indices"] = [0]
    candidate = tmp_path / "watch.jsonl"
    candidate.write_text(json.dumps(altered) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        verify(HERE / "receipt.json", candidate)

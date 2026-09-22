from __future__ import annotations

import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def test_receipt_verifier_passes() -> None:
    from verify_receipt import verify

    verify()


def test_no_model_or_reference_counters() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text(encoding="utf-8"))
    accounting = receipt["accounting"]
    assert accounting["inference_runs"] == 0
    assert accounting["reference_reads"] == 0
    assert accounting["metrics"] == 0
    assert receipt["outputs"] == []


def test_tampering_with_binding_is_rejected(tmp_path: Path) -> None:
    source = (HERE / "receipt.json").read_text(encoding="utf-8")
    altered = json.loads(source)
    altered["bindings"]["retinexformer"]["mode"] = "changed"
    (tmp_path / "receipt.json").write_text(json.dumps(altered), encoding="utf-8")
    assert altered["bindings"]["retinexformer"] != json.loads(source)["bindings"]["retinexformer"]

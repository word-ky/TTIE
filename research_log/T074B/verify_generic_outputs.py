"""Independent verifier for T074-B generic low-only output rows (no reference access, no metrics)."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

KINDS = ("retinexformer", "snr_aware", "promptir", "promptir_dctta", "ours_step0", "ours_ttt")
ZERO_STATE_SHA256 = hashlib.sha256(torch.zeros(1, 3, 2, 2).numpy().tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def verify_row(row, item, root):
    shape = [1, 3, item["height"], item["width"]]
    assert row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"]
    assert row["shape"] == shape and row["dtype"] == "torch.float32"
    directory = root / Path(item["name"]).stem
    assert json.loads((directory / "decision.json").read_bytes()) == row, "decision.json differs from manifest row"
    output = directory / "output.pt.gz"
    assert sha256_file(output) == row["output_file_sha256"]
    tensor = load_tensor(output)
    assert list(tensor.shape) == shape and tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == row["output_tensor_sha256"]
    return output.stat().st_size


def verify_abstention(row, step0_row):
    assert row["low_name"] == step0_row["low_name"] and row["low_sha256"] == step0_row["low_sha256"]
    if not any(step0_row["active"]):
        assert row["decision_status"] == row["decision"]["status"] == "TTT_ABSTAIN_NO_ACTIVE_GATE"
        assert row["decision"]["selected_step"] == 0 and row["optimizer_updates"] == 0
        assert row["active"] == step0_row["active"]
        assert row["output_tensor_sha256"] == step0_row["output_tensor_sha256"]
        assert row["output_file_sha256"] == step0_row["output_file_sha256"]
        assert row["selected_state_sha256"] == ZERO_STATE_SHA256
        return True
    assert row["decision_status"] == "TTT_EXECUTED"
    assert 0 <= row["decision"]["selected_step"] <= 27
    assert len(row["selected_state_sha256"]) == 64
    return False


def verify(kind, low_receipt, outputs, expected_count, step0_outputs=None, execution_manifest=None):
    assert kind in KINDS
    files = json.loads(Path(low_receipt).read_bytes())["files"]
    manifest_path = Path(outputs) / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    names = [item["name"] for item in files]
    assert len(names) == len(set(names)) == expected_count
    assert manifest["count"] == len(manifest["rows"]) == expected_count
    assert manifest["low_receipt_sha256"] == sha256_file(low_receipt)
    assert manifest["reference_reads"] == 0 and manifest["metrics"] == 0
    assert manifest.get("promotable", True) is True, "smoke manifests are not promotable"
    if execution_manifest is not None:
        assert manifest["execution_manifest_sha256"] == sha256_file(execution_manifest)
    total, abstained = 0, 0
    step0 = None
    if kind == "ours_ttt":
        step0_path = Path(step0_outputs) / "output_manifest.json"
        step0 = json.loads(step0_path.read_bytes())
        assert manifest["step0_output_manifest_sha256"] == sha256_file(step0_path)
        assert step0["low_receipt_sha256"] == manifest["low_receipt_sha256"]
        assert step0["execution_manifest_sha256"] == manifest["execution_manifest_sha256"]
        assert step0["count"] == len(step0["rows"]) == expected_count
    for index, (item, row) in enumerate(zip(files, manifest["rows"])):
        total += verify_row(row, item, Path(outputs))
        if kind in ("ours_step0", "ours_ttt"):
            assert row["execution_manifest_sha256"] == manifest["execution_manifest_sha256"]
        if kind == "ours_step0":
            assert len(row["active"]) == 4 and all(isinstance(v, bool) for v in row["active"])
        if kind == "ours_ttt":
            abstained += verify_abstention(row, step0["rows"][index])
    if kind == "promptir_dctta":
        assert manifest["expected_order_sha256"] and manifest["adaptation_scope"] == "all_target_low_domain_level"
        assert manifest["parameter_updates"]["changed_elements"] > 0
        assert not manifest["parameter_updates"]["changed_outside_trainable"]
    if kind == "snr_aware":
        assert manifest["parameter_hash_before"] == manifest["parameter_hash_after"]
    if kind in ("promptir", "promptir_dctta"):
        assert manifest["model_state_unchanged"] is True
    return {
        "classification": f"T074B_{kind.upper()}_OUTPUTS_VERIFIED",
        "count": expected_count,
        "output_manifest_sha256": sha256_file(manifest_path),
        "low_receipt_sha256": manifest["low_receipt_sha256"],
        "compressed_output_bytes": total,
        **({"ttt_abstain_no_active_gate_count": abstained,
            "ttt_executed_count": expected_count - abstained} if kind == "ours_ttt" else {}),
        "reference_reads": 0,
        "metrics": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--step0-outputs", type=Path)
    p.add_argument("--execution-manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    receipt = verify(a.kind, a.low_receipt, a.outputs, a.expected_count, a.step0_outputs, a.execution_manifest)
    with open(a.out, "x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""T074-C phase-3 reference gate (no target GT is opened, hashed or decoded here).

For one target, re-verify every required non-tuned row:
freeze receipt -> output manifest SHA -> verification SHA, on the local (committed) JSONs and on the
remote run directory (remote ``output_manifest.json``, ``<remote_output_root>.verify.json`` and every
``<stem>/output.pt.gz``). Any missing / pending / inconsistent row fails closed. Only when every required
row passes is the immutable ``reference_gate_receipt.json`` written (``open(..., "x")``, then read-only).
Target GT may be decoded only by code that first calls ``validate_receipt`` on that receipt.

Checks use explicit ``GateError`` raises, never ``assert`` (which ``python -O`` would strip).
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSIFICATION = "T074C_REFERENCE_GATE_OPEN"
TUNED_ROW = "ours_ttt_target_tuned"

# Registry of required rows (preregistration research_log/T074C/prereg.md, section "Rows").
# The tuned row is produced after the gate and is never a gate input.
TARGETS = {
    "SDSD_indoor": {
        "display_name": "SDSD-indoor",
        "dataset": "sdsd_in",
        "geometry": "snr960x512",
        "expected_count": 180,
        "low_receipt_sha256": "19138a9c1fef4e1d3a6ffcf6f0bf9dcb3b5b64453682f7424a3ea5f749512978",
        "reference_opaque_manifest_sha256": "dd7415dd7a8f212eeed8d7adf23e2948c0687937c63a5384f634ead655d722a2",
        "stage_script_sha256": "7f6ff8bc901eee6cbce987535f9197a6342442788883e6dbe1c7ae9e23c9b902",
        "required_rows": [
            "retinexformer", "snr_aware", "promptir", "promptir_dctta",
            "mr_illuminate", "quadprior", "ours_step0", "ours_ttt",
        ],
    },
}


class GateError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    """(object, sha256) of the same bytes: no gap between hashing and parsing."""
    path = Path(path)
    require(path.is_file(), f"missing file {path}")
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def utc():
    return datetime.now(timezone.utc).isoformat()


def target_spec(target):
    require(target in TARGETS, f"unknown target {target!r}; registry has {sorted(TARGETS)}")
    return TARGETS[target]


# ---------------------------------------------------------------- inputs shared with metrics


def check_low_and_opaque(spec, low_receipt_path, opaque_path):
    low, low_sha = load_json(low_receipt_path)
    require(low_sha == spec["low_receipt_sha256"], f"low receipt SHA {low_sha} != registry")
    rule = low["rule"]
    require(rule["dataset"] == spec["dataset"] and rule["geometry"] == spec["geometry"], "low receipt rule mismatch")
    require(rule["stage_script_sha256"] == spec["stage_script_sha256"], "low receipt stager hash mismatch")
    require(low["count"] == len(low["files"]) == spec["expected_count"], "low receipt count mismatch")
    require(low["reference_reads"] == 0 and low["reference_decodes"] == 0 and low["metrics"] == 0,
            "low receipt records reference access")
    names = [f["name"] for f in low["files"]]
    require(len(set(names)) == len(names) and names == sorted(names), "low receipt names not unique/sorted")
    opaque, opaque_sha = load_json(opaque_path)
    require(opaque_sha == spec["reference_opaque_manifest_sha256"], f"reference-opaque manifest SHA {opaque_sha} != registry")
    require(opaque["low_receipt_sha256"] == low_sha, "opaque manifest bound to a different low receipt")
    require(opaque["rule"] == rule, "opaque manifest rule differs from low receipt rule")
    require(opaque["reference_decodes"] == 0, "opaque manifest records a reference decode")
    require(opaque["count"] == len(opaque["pairs"]) == len(names), "opaque manifest count mismatch")
    for item, pair in zip(low["files"], opaque["pairs"]):
        require(pair["name"] == item["name"] and pair["cluster"] == item["cluster"],
                f"opaque/low pairing mismatch at {item['name']}")
    return low, low_sha, opaque, opaque_sha


# ---------------------------------------------------------------- rows


def _check_manifest_rows(row_id, manifest, low_files, low_sha):
    require(manifest["count"] == len(manifest["rows"]) == len(low_files), f"{row_id}: manifest count mismatch")
    require(manifest["low_receipt_sha256"] == low_sha, f"{row_id}: manifest bound to another low receipt")
    require(manifest["reference_reads"] == 0 and manifest["metrics"] == 0, f"{row_id}: manifest records reference/metric access")
    require(manifest.get("promotable", True) is True, f"{row_id}: manifest is not promotable")
    for item, row in zip(low_files, manifest["rows"]):
        require(row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"],
                f"{row_id}: manifest row/low mismatch at {item['name']}")
        require(row["shape"] == [1, 3, item["height"], item["width"]] and row["dtype"] == "torch.float32",
                f"{row_id}: output geometry/dtype mismatch at {item['name']}")


def check_row(row_id, rows_dir, spec, low_files, low_sha, verify_outputs):
    directory = Path(rows_dir) / row_id
    freeze_path = directory / "freeze_receipt.json"
    if not freeze_path.is_file():
        status = directory / "status.json"
        detail = json.loads(status.read_bytes()).get("classification") if status.is_file() else "no freeze receipt"
        raise GateError(f"{row_id}: required row not frozen ({detail})")
    freeze, freeze_sha = load_json(freeze_path)
    require(freeze["classification"] == "FROZEN_OUTPUTS", f"{row_id}: classification {freeze['classification']}")
    require(freeze["method_id"] == row_id and freeze["target"] == spec["display_name"], f"{row_id}: freeze identity mismatch")
    require(freeze["output_count"] == spec["expected_count"], f"{row_id}: output_count mismatch")
    require(freeze["canonical_low_receipt_sha256"] == low_sha, f"{row_id}: freeze bound to another low receipt")
    require(freeze["target_reference_reads"] == 0 and freeze["target_metrics"] == 0,
            f"{row_id}: freeze records reference/metric access")
    require(freeze["preregistration_changed"] is False and freeze["quality_inspection_of_outputs"] is False,
            f"{row_id}: freeze records a preregistration change or output inspection")

    manifest, manifest_sha = load_json(directory / "output_manifest.json")
    require(manifest_sha == freeze["output_manifest_sha256"], f"{row_id}: local output manifest SHA mismatch")
    _check_manifest_rows(row_id, manifest, low_files, low_sha)
    verification, verification_sha = load_json(directory / "verification.json")
    require(verification_sha == freeze["independent_verification_sha256"], f"{row_id}: local verification SHA mismatch")
    require(verification["classification"] == freeze["verification_classification"], f"{row_id}: verification classification")
    require(verification["output_manifest_sha256"] == manifest_sha, f"{row_id}: verification bound to another manifest")
    require(verification["low_receipt_sha256"] == low_sha, f"{row_id}: verification bound to another low receipt")
    require(verification["reference_reads"] == 0 and verification["metrics"] == 0,
            f"{row_id}: verification records reference/metric access")

    remote = Path(freeze["remote_output_root"])
    require(sha256_file(remote / "output_manifest.json") == manifest_sha, f"{row_id}: remote output manifest SHA mismatch")
    remote_verify = Path(str(remote) + ".verify.json")
    require(remote_verify.is_file() and sha256_file(remote_verify) == verification_sha,
            f"{row_id}: remote verification SHA mismatch")
    if verify_outputs:
        for item, row in zip(low_files, manifest["rows"]):
            output = remote / Path(item["name"]).stem / "output.pt.gz"
            require(output.is_file() and sha256_file(output) == row["output_file_sha256"],
                    f"{row_id}: output file SHA mismatch for {item['name']}")
    return {
        "freeze_receipt_sha256": freeze_sha,
        "output_manifest_sha256": manifest_sha,
        "verification_sha256": verification_sha,
        "verification_classification": verification["classification"],
        "remote_output_root": str(remote),
    }, freeze, manifest


def check_cross_rows(rows_dir, freezes, manifests, entries):
    if "ours_ttt" in freezes and "ours_step0" in freezes:
        step0 = entries["ours_step0"]["output_manifest_sha256"]
        require(freezes["ours_ttt"]["expected_step0_manifest_sha256"] == step0, "ours_ttt: bound to another Step0 manifest")
        require(manifests["ours_ttt"]["step0_output_manifest_sha256"] == step0, "ours_ttt: manifest Step0 binding mismatch")
    if "promptir_dctta" in freezes:
        freeze = freezes["promptir_dctta"]
        _, order_sha = load_json(Path(rows_dir) / "promptir_dctta" / "adaptation_order.json")
        require(order_sha == freeze["sealed_adaptation_order_sha256"] == manifests["promptir_dctta"]["expected_order_sha256"],
                "promptir_dctta: sealed adaptation order mismatch")


def evaluate(target, rows_dir, low_receipt_path, opaque_path, verify_outputs=True):
    """All gate checks; collects every row failure before failing closed. Returns receipt fields."""
    spec = target_spec(target)
    low, low_sha, opaque, opaque_sha = check_low_and_opaque(spec, low_receipt_path, opaque_path)
    entries, freezes, manifests, failures = {}, {}, {}, []
    for row_id in spec["required_rows"]:
        try:
            entries[row_id], freezes[row_id], manifests[row_id] = check_row(
                row_id, rows_dir, spec, low["files"], low_sha, verify_outputs)
        except (GateError, KeyError, OSError, ValueError) as error:
            failures.append(f"{row_id}: {type(error).__name__}: {error}")
    if failures:
        raise GateError("reference gate closed; " + " | ".join(failures))
    check_cross_rows(rows_dir, freezes, manifests, entries)
    return {
        "task": "T074-C",
        "classification": CLASSIFICATION,
        "target": target,
        "display_name": spec["display_name"],
        "gate_code_sha256": sha256_file(__file__),
        "low_receipt_sha256": low_sha,
        "reference_opaque_manifest_sha256": opaque_sha,
        "stage_script_sha256": spec["stage_script_sha256"],
        "count": low["count"],
        "cluster_count": len({f["cluster"] for f in low["files"]}),
        "required_rows": list(spec["required_rows"]),
        "rows": entries,
        "ours_ttt_no_active_abstentions": freezes.get("ours_ttt", {}).get("ttt_abstain_no_active_gate_count"),
        "target_reference_reads_before_gate": 0,
        "target_metrics_before_gate": 0,
    }


def write_receipt(fields, out):
    receipt = dict(fields, created_utc=utc(), output_files_rehashed=True)
    with open(out, "x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(out, 0o444)
    return receipt


def validate_receipt(receipt_path, target, rows_dir, low_receipt_path, opaque_path):
    """Refuse unless an intact gate receipt exists and every bound file is unchanged. Returns (receipt, sha)."""
    receipt_path = Path(receipt_path)
    require(receipt_path.is_file(), f"no reference gate receipt at {receipt_path}; GT decoding refused")
    receipt, receipt_sha = load_json(receipt_path)
    require(receipt.get("classification") == CLASSIFICATION and receipt.get("target") == target,
            "reference gate receipt classification/target mismatch")
    require(receipt.get("output_files_rehashed") is True and "created_utc" in receipt, "incomplete gate receipt")
    current = evaluate(target, rows_dir, low_receipt_path, opaque_path, verify_outputs=False)
    for key, value in current.items():
        require(receipt.get(key) == value, f"gate receipt field {key!r} no longer matches the bound files/code")
    return receipt, receipt_sha


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--rows-dir", type=Path, help="default: research_log/T074C/<target>")
    p.add_argument("--out", type=Path, help="default: <rows-dir>/reference_gate_receipt.json")
    a = p.parse_args()
    rows_dir = a.rows_dir or HERE / a.target
    out = a.out or rows_dir / "reference_gate_receipt.json"
    require(not out.exists(), f"{out} exists; the gate receipt is immutable")
    receipt = write_receipt(evaluate(a.target, rows_dir, a.low_receipt, a.opaque_manifest), out)
    print(json.dumps({"receipt": str(out), "sha256": sha256_file(out), "rows": receipt["required_rows"]}, indent=2))


if __name__ == "__main__":
    main()

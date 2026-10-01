"""Low-only SMID default-output reproduction check on migrated A6000 GPU 0."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/TTIE")
SOURCE = ROOT / "T073C/recovery/source"
GATE_TREE = ROOT / "T075A/gatetree"
T074C = GATE_TREE / "research_log/T074C"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(T074C))
sys.path.insert(0, str(GATE_TREE / "research_log/T075A/v2"))

import metrics
import ours_tuning as tuning
import v2_gate_metrics as gate_metrics
from research_log.T070A.infer import FinalOurs
from research_log.T070A.manifest import environment
from ttie.lolv2_gamma_core import native_rgb


parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true", help="check all 1470 frozen low inputs")
args = parser.parse_args()

target = ROOT / "T074B/targets/SMID"
rows_dir = GATE_TREE / "research_log/T075B/SMID"
gate_metrics.register("SMID")
ctx = metrics.Context(
    "SMID",
    target / "low_receipt.json",
    target / "reference_opaque_manifest.json",
    rows_dir / "reference_gate_receipt.json",
    rows_dir,
)
manifest = ROOT / "T073C/recovery/artifacts/T073C_execution_manifest.json"
original_sha = tuning.sha256_file(manifest)
diagnostic_manifest = ROOT / "diagnostic_manifest_A6000.json"
description = json.loads(manifest.read_bytes())
description["environment"] = environment()
diagnostic_manifest.write_text(json.dumps(description, indent=2))
model = FinalOurs(diagnostic_manifest)
frozen, frozen_sha = tuning.load_json(rows_dir / "ours_ttt/output_manifest.json")
assert frozen_sha == ctx.receipt["rows"]["ours_ttt"]["output_manifest_sha256"]
assert frozen["execution_manifest_sha256"] == original_sha

knobs = tuning.canonical_knobs({})
variant = {
    "id": "default",
    "knobs": knobs,
    "derived": tuning.derive(knobs, model.manifest["source_binding"])[0],
}
items = ctx.files if args.all else ctx.files[:1]
for index, item in enumerate(items):
    low_path = target / "low" / item["name"]
    assert tuning.sha256_file(low_path) == item["sha256"]
    low = native_rgb(low_path)
    image, decision = tuning.run_group(
        model.scorer, model.gate, model.model, low, [variant], "cuda:0"
    )["default"]
    actual = None if image is None else tuning.sha256_bytes(image.contiguous().numpy().tobytes())
    expected = frozen["rows"][index]["output_tensor_sha256"]
    if actual != expected:
        print(json.dumps({
            "target": "SMID", "index": index, "name": item["name"],
            "expected_tensor_sha256": expected, "actual_tensor_sha256": actual,
            "decision": decision, "reference_reads": len(ctx.reads),
        }, indent=2, default=str))
        raise SystemExit(2)
    if (index + 1) % 100 == 0 or index + 1 == len(items):
        print(f"BITWISE_MATCH {index + 1}/{len(items)}", flush=True)

print(json.dumps({
    "target": "SMID", "count": len(items), "bitwise_equal": True,
    "gate_receipt_sha256": ctx.receipt_sha256,
    "frozen_execution_manifest_sha256": original_sha,
    "diagnostic_execution_manifest_sha256": model.manifest_sha256,
    "reference_reads": len(ctx.reads),
}, indent=2))

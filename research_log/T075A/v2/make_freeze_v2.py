"""Freeze receipt for one verified T075-A v2 row, in the T074-C / T075-B key format (no reference access, no metrics).

usage: make_freeze_v2.py --target LSRW --row ours_v2 --outputs <run dir> --source-freeze <source freeze_receipt.json>
                         --out <freeze dir> --path-prefix research_log/T075B/LSRW/ours_v2
Writes <out>/{freeze_receipt.json, output_manifest.json, verification.json}; the two JSONs are byte copies of
<outputs>/output_manifest.json and <outputs>.verify.json (the gate re-hashes both copies and both remote files).
"""

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

DISPLAY = {"LSRW": "LSRW", "SMID": "SMID", "SDSD_indoor": "SDSD-indoor"}
KINDS = {"ours_ttt_sdsd_knobs": "knobs"}
RUNNER = {"knobs": "research_log/T075A/v2/run_ours_knobs.py", "plus_d": "research_log/T075A/v2/denoise_d.py"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def make(target, row, outputs, source_freeze, out, prefix):
    outputs, out = Path(outputs), Path(out)
    kind = KINDS.get(row, "plus_d")
    need(not out.exists(), f"{out} exists")
    verify = Path(str(outputs) + ".verify.json")
    man = json.loads((outputs / "output_manifest.json").read_bytes())
    ver = json.loads(verify.read_bytes())
    src = json.loads(Path(source_freeze).read_bytes())
    need(man["method_id"] == ver["method_id"] == row, "row identity")
    need(man["reference_reads"] == 0 and man["metrics"] == 0 and ver["reference_reads"] == 0 and ver["metrics"] == 0,
         "reference/metric access recorded")
    need(ver["classification"] == f"T075A_{kind.upper()}_OUTPUTS_VERIFIED", "verification did not pass")
    need(ver["output_manifest_sha256"] == sha(outputs / "output_manifest.json"), "verification bound to another manifest")
    need(src["classification"] == "FROZEN_OUTPUTS" and src["target"] == DISPLAY.get(target, target), "source freeze identity")
    need(src["canonical_low_receipt_sha256"] == man["low_receipt_sha256"], "source freeze bound to another low receipt")
    out.mkdir(parents=True)
    shutil.copyfile(outputs / "output_manifest.json", out / "output_manifest.json")
    shutil.copyfile(verify, out / "verification.json")
    rows = man["rows"]
    secs = [r["whole_run_seconds"] for r in rows]
    receipt = {
        "task": "T075-A-v2", "method_id": row, "target": DISPLAY.get(target, target), "classification": "FROZEN_OUTPUTS",
        "output_count": len(rows), "preregistration": "research_log/T075A/v2_freeze.md",
        "canonical_low_receipt_sha256": man["low_receipt_sha256"],
        "runner": RUNNER[kind], "runner_sha256": man["producer_sha256"],
        "source_row": src["method_id"], "source_freeze_receipt_sha256": sha(source_freeze),
        "source_output_manifest_sha256": src["output_manifest_sha256"],
        "verifier": "research_log/T075A/v2/verify_v2_rows.py", "verifier_sha256": ver["verifier_sha256"],
        "output_manifest_path": f"{prefix}/output_manifest.json", "output_manifest_sha256": sha(out / "output_manifest.json"),
        "independent_verification_path": f"{prefix}/verification.json",
        "independent_verification_sha256": sha(out / "verification.json"),
        "verification_classification": ver["classification"],
        "per_image_seconds_median": statistics.median(secs), "per_image_seconds_sum": sum(secs),
        "max_per_image_peak_reserved_gpu_bytes": max(r["peak_gpu_memory_bytes"] for r in rows),
        "compressed_output_bytes": ver["compressed_output_bytes"], "remote_output_root": str(outputs),
        "target_reference_reads": 0, "target_metrics": 0, "quality_inspection_of_outputs": False,
        "preregistration_changed": False,
    }
    if kind == "plus_d":
        need(man["source_row"] == src["method_id"] and man["source_output_manifest_sha256"] == src["output_manifest_sha256"],
             "row not derived from the given frozen source row")
        receipt.update({"d_spec": man["d_spec"], "recomputed_images": ver["recomputed_images"],
                        "process_env": "CPU only; OpenCV 1 thread per worker"})
    else:
        need(src["method_id"] == "ours_ttt" and man["frozen_ours_ttt_manifest_sha256"] == src["output_manifest_sha256"],
             "knobs row not bound to the frozen ours_ttt row")
        receipt.update({"execution_manifest_sha256": man["execution_manifest_sha256"], "knobs": man["knobs"],
                        "setting_id": man["setting_id"], "default_reproduction_images": man["default_reproduction_images"],
                        "ttt_abstain_no_active_gate_count": ver["no_active_abstentions"],
                        "ttt_executed_count": ver["ttt_executed_count"],
                        "process_env": "CUBLAS_WORKSPACE_CONFIG, OMP/MKL/OPENBLAS_NUM_THREADS, MKL_THREADING_LAYER unset",
                        "gpu_serialization": "flock /root/autodl-tmp/TTIE/gpu.lock"})
    (out / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target", choices=sorted(DISPLAY), required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--source-freeze", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--path-prefix", required=True)
    a = p.parse_args()
    r = make(a.target, a.row, a.outputs, a.source_freeze, a.out, a.path_prefix)
    print(json.dumps({"row": a.row, "manifest": r["output_manifest_sha256"], "verification": r["independent_verification_sha256"],
                      "freeze": sha(a.out / "freeze_receipt.json"), "median_s": round(r["per_image_seconds_median"], 3)}))


if __name__ == "__main__":
    main()

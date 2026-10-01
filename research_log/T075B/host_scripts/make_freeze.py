"""T075-B: build a freeze receipt (T074-C key format) for one verified row from host run artifacts.

usage: make_freeze.py DS ROW   -> /root/autodl-tmp/TTIE/T075B/freeze/<DS>/<ROW>/{freeze_receipt.json, output_manifest.json,
       verification.json[, adaptation_order.json]} (byte copies of the remote files; receipt hashes the copies).
No reference access, no metrics.
"""
import hashlib, json, shutil, statistics, sys
from pathlib import Path

TC = Path("/root/autodl-tmp/TTIE/T075B")
DISPLAY = {"LSRW": "LSRW", "SMID": "SMID"}
DIRS = {"ours_step0": "ours_step0", "ours_ttt": "ours_ttt", "retinexformer": "retinexformer", "snr_aware": "snr_aware",
        "promptir": "promptir", "promptir_dctta": "dctta", "quadprior": "quadprior", "mr_illuminate": "mri"}
SMOKE = {"ours_step0": "ours_step0_smoke", "ours_ttt": "ours_ttt_smoke", "retinexformer": "retinexformer_smoke",
         "snr_aware": "snr_aware_smoke", "promptir": "promptir_smoke", "promptir_dctta": "dctta_smoke",
         "quadprior": "quadprior_smoke", "mr_illuminate": "mri_smoke"}
CODE = {
    "ours_step0": "/root/autodl-tmp/TTIE/T074B/code/run_ours_step0_generic.py",
    "ours_ttt": "/root/autodl-tmp/TTIE/T074B/code/run_ours_ttt_abstain_generic.py",
    "retinexformer": "/root/autodl-tmp/TTIE/T074B/code/run_retinex_snr_generic.py",
    "snr_aware": "/root/autodl-tmp/TTIE/T074B/code/run_retinex_snr_generic.py",
    "promptir": "/root/autodl-tmp/TTIE/T074B/code/run_promptir_static_generic.py",
    "promptir_dctta": "/root/autodl-tmp/TTIE/T074B/code/run_dctta_generic.py",
    "quadprior": "/root/autodl-tmp/TTIE/T073E/code/run_quadprior_batch.py",
    "mr_illuminate": "/root/autodl-tmp/TTIE/T073D/code/run_mri_native_batch.py",
}
LOCAL = {"ours_step0": "research_log/T074B/run_ours_step0_generic.py", "ours_ttt": "research_log/T074B/run_ours_ttt_abstain_generic.py",
         "retinexformer": "research_log/T074B/run_retinex_snr_generic.py", "snr_aware": "research_log/T074B/run_retinex_snr_generic.py",
         "promptir": "research_log/T074B/run_promptir_static_generic.py", "promptir_dctta": "research_log/T074B/run_dctta_generic.py",
         "quadprior": "research_log/T073E/run_quadprior_batch.py", "mr_illuminate": "research_log/T073D/run_mri_native_batch.py"}
VERIFIER = {"quadprior": "/root/autodl-tmp/TTIE/T073E/code/verify_quadprior_full.py",
            "mr_illuminate": "/root/autodl-tmp/TTIE/T073D/code/verify_mri_full.py"}
VERIFIER_LOCAL = {"quadprior": "research_log/T073E/verify_quadprior_full.py", "mr_illuminate": "research_log/T073D/verify_mri_full.py"}
GENERIC_VERIFIER = "/root/autodl-tmp/TTIE/T074B/code/verify_generic_outputs.py"
CLEAN = "CUBLAS_WORKSPACE_CONFIG, OMP/MKL/OPENBLAS_NUM_THREADS, MKL_THREADING_LAYER unset (as frozen UHD-LL runs)"
AZENV = "CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL/OPENBLAS_NUM_THREADS=1 (as T072-AZ)"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ds, row = sys.argv[1], sys.argv[2]
    runs = TC / "runs" / ds
    d = runs / DIRS[row]
    out = TC / "freeze" / ds / row
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(d / "output_manifest.json", out / "output_manifest.json")
    shutil.copyfile(runs / f"{DIRS[row]}.verify.json", out / "verification.json")
    man = json.loads((out / "output_manifest.json").read_bytes())
    ver = json.loads((out / "verification.json").read_bytes())
    smoke_sha = sha_file(runs / SMOKE[row] / "output_manifest.json")
    rows = man["rows"]
    secs = [r.get("whole_run_seconds", 0) for r in rows]
    peaks = [r.get("peak_gpu_memory_bytes", 0) for r in rows]
    scalars = {k: v for k, v in man.items() if not isinstance(v, (list, dict)) and k not in ("low_dir", "workdir")}
    extra = {"runner": LOCAL[row], "runner_sha256": sha_lf(CODE[row]), "smoke_manifest_sha256": smoke_sha,
             "smoke_promotable": False, "manifest_scalars": scalars, "gpu_serialization": "flock /root/autodl-tmp/TTIE/gpu.lock"}
    if row in ("ours_step0", "ours_ttt", "promptir", "promptir_dctta"):
        extra["process_env"] = CLEAN
    if row in ("retinexformer", "snr_aware"):
        extra["process_env"] = AZENV
    if row == "ours_step0":
        extra["gate_active_images"] = sum(any(r["active"]) for r in rows)
        extra["gate_inactive_images"] = sum(not any(r["active"]) for r in rows)
    if row == "ours_ttt":
        extra["expected_step0_manifest_sha256"] = man["step0_output_manifest_sha256"]
        extra["ttt_abstain_no_active_gate_count"] = ver["ttt_abstain_no_active_gate_count"]
        extra["ttt_executed_count"] = ver["ttt_executed_count"]
    if row == "promptir_dctta":
        shutil.copyfile(runs / "dctta_order" / "adaptation_order.json", out / "adaptation_order.json")
        order_sha = sha_file(out / "adaptation_order.json")
        assert order_sha == man["expected_order_sha256"]
        extra.update({"sealed_adaptation_order_path": f"research_log/T075B/{ds}/promptir_dctta/adaptation_order.json",
                      "sealed_adaptation_order_sha256": order_sha, "parameter_updates": {
                          k: v for k, v in man["parameter_updates"].items() if k != "changed_tensor_names"},
                      "reproducibility_note": "single-shot; adapted state not bitwise reproducible across reruns; must not be rerun"})
    if row in ("quadprior", "mr_illuminate"):
        extra["weights"] = man.get("weights") or man.get("vae_checkpoint")
        extra["model_state_sha256"] = man.get("model_state_sha256")
        extra["first_row_frozen_smoke_verified"] = ver.get("first_row_frozen_smoke_verified")
        extra["verifier_pass_count"] = ver.get("pass_count")
        if row == "mr_illuminate":
            extra["sd15_revision"] = man["sd_snapshot"]["revision"]
            extra["environment_version_check"] = man.get("environment_version_check")
    extra["verifier"] = VERIFIER_LOCAL.get(row, "research_log/T074B/verify_generic_outputs.py")
    extra["verifier_sha256"] = sha_lf(VERIFIER.get(row, GENERIC_VERIFIER))
    receipt = {
        "task": "T075-B", "method_id": row, "target": DISPLAY[ds], "classification": "FROZEN_OUTPUTS",
        "output_count": len(rows), "preregistration": "research_log/T074C/prereg.md + research_log/T075A/plan_and_amendments.md",
        "canonical_low_receipt_sha256": man["low_receipt_sha256"], **extra,
        "output_manifest_path": f"research_log/T075B/{ds}/{row}/output_manifest.json",
        "output_manifest_sha256": sha_file(out / "output_manifest.json"),
        "independent_verification_path": f"research_log/T075B/{ds}/{row}/verification.json",
        "independent_verification_sha256": sha_file(out / "verification.json"),
        "verification_classification": ver["classification"],
        "per_image_seconds_median": statistics.median(secs), "per_image_seconds_sum": sum(secs),
        "max_per_image_peak_reserved_gpu_bytes": max(peaks),
        "compressed_output_bytes": ver.get("compressed_output_bytes"),
        "remote_output_root": str(d), "target_reference_reads": 0, "target_metrics": 0,
        "quality_inspection_of_outputs": False, "preregistration_changed": False,
    }
    assert man["reference_reads"] == 0 and man["metrics"] == 0 and ver["reference_reads"] == 0 and ver["metrics"] == 0
    assert ver["output_manifest_sha256"] == receipt["output_manifest_sha256"]
    assert ver["classification"].endswith("_OUTPUTS_VERIFIED")
    assert sha_file(Path(str(d) + ".verify.json")) == receipt["independent_verification_sha256"]
    (out / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"row": row, "manifest": receipt["output_manifest_sha256"], "verification": receipt["independent_verification_sha256"],
                      "freeze": sha_file(out / "freeze_receipt.json"), "median_s": round(receipt["per_image_seconds_median"], 3),
                      "sum_s": round(sum(secs), 1), "max_peak_gib": round(max(peaks) / 2**30, 3),
                      **({"abstain": extra["ttt_abstain_no_active_gate_count"], "executed": extra["ttt_executed_count"]} if row == "ours_ttt" else {}),
                      **({"order": extra["sealed_adaptation_order_sha256"], "adapt_s": man.get("adaptation_seconds"),
                          "adapt_peak": man.get("adaptation_peak_gpu_memory_bytes")} if row == "promptir_dctta" else {})}))


if __name__ == "__main__":
    main()

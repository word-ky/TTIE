"""One authorized SNR native-4K smoke, then 150+150 full runs if it passes."""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TTIE/T072AZ")
EV = ROOT / "T072AZ"
HERE = EV / "continuation"
RUNS = Path("/root/autodl-fs/TTIE/T072AZ/runs")
RUNTIME = ROOT / "runtime"
STATUS = RUNS / "continuation_status.json"
sys.path.insert(0, str(EV / "seal"))
from launcher import RealBackend, choose


def save(obj):
    STATUS.write_text(json.dumps(obj, indent=2) + "\n")


def run(args, log):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log.open("x") as f:
        return subprocess.run(args, cwd=RUNTIME, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    status = {"task": "T072-AZ-continuation", "classification": "STARTED",
              "reference_reads": 0, "metrics": 0, "snr_smoke_count": 0, "full": []}
    save(status)
    try:
        backend = RealBackend(RUNTIME)
        snapshot = backend.snapshot()
        (RUNS / "continuation_gpu_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        device = choose(snapshot)
        if device is None:
            status["classification"] = "BLOCKED_GPU_GATE"
            save(status)
            return
        spec = json.loads((EV / "seal/spec.json").read_text())
        bindings = json.loads((EV / "seal/baseline_bindings.json").read_text())
        backend.bindings(spec, bindings)
        backend.input(spec["smoke"])
        prior = json.loads((RUNS / "smoke/receipt.json").read_text())
        assert prior["runs"][0]["method"] == "retinexformer" and prior["runs"][0]["status"] == "FROZEN"
        smoke = RUNS / "continuation_snr_smoke"
        smoke.mkdir(exist_ok=False)
        status["snr_smoke_count"] = 1
        save(status)
        ret = run([sys.executable, str(HERE / "smoke_worker.py"), "snr_aware", str(smoke), str(RUNTIME)],
                  smoke / "process.log")
        status["snr_smoke_returncode"] = ret
        save(status)
        if ret:
            status["classification"] = "BLOCKED_NATIVE4K_SNR_AWARE_HARDWARE"
            save(status)
            return
        result = json.loads((smoke / "output_freeze.json").read_text())
        assert result["shape"] == [2160, 3840, 3] and result["dtype"] == "float32" and result["finite"] is True
        status["snr_smoke"] = result
        status["classification"] = "SNR_SMOKE_PASS_FULL_IN_PROGRESS"
        save(status)
        lows = json.loads((EV / "low_receipt.json").read_text())["files"]
        paths = [str(ROOT / "input/T072I_input_cache" / row["name"]) for row in lows]
        for method in ("retinexformer", "snr_aware"):
            b = bindings[method]
            out = RUNS / "full" / method
            if method == "retinexformer":
                args = [sys.executable, "-m", "ttie.retinex_exporter", "--low", *paths,
                        "--checkpoint", b["checkpoint"], "--config", b["config"], "--out", str(out)]
            else:
                args = [sys.executable, str(HERE / "full_worker.py"), "--low", *paths,
                        "--checkpoint", b["checkpoint"], "--config", b["config"],
                        "--out", str(out), "--source", b["source"]]
            status["full"].append({"method": method, "count": 150, "status": "STARTED"})
            save(status)
            ret = run(args, RUNS / ("continuation_" + method + ".log"))
            status["full"][-1].update(returncode=ret, status="PROCESS_DONE" if ret == 0 else "FAILED")
            save(status)
            if ret:
                status["classification"] = "BLOCKED_FULL_" + method.upper()
                save(status)
                return
        status["classification"] = "FULL_PROCESSES_DONE_AWAITING_INDEPENDENT_VERIFY"
        save(status)
        ret = run([sys.executable, str(HERE / "verify_full.py")], RUNS / "continuation_verify.log")
        status["verification_returncode"] = ret
        status["classification"] = "UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090" if ret == 0 else "BLOCKED_FULL_VERIFICATION"
    except Exception as e:
        status.update(classification="BLOCKED_CONTINUATION_EXECUTION", error_type=type(e).__name__,
                      error=str(e), traceback=traceback.format_exc())
    save(status)


if __name__ == "__main__":
    main()

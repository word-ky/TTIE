"""One T072-AZ smoke pair, then one complete frozen low-only pass per baseline."""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/TTIE/T072AZ")
EVIDENCE = ROOT / "T072AZ"
RUNTIME = ROOT / "runtime"
RUNS = Path("/root/autodl-fs/TTIE/T072AZ/runs")
METHODS = ("retinexformer", "snr_aware")


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def command(args, cwd, log):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with log.open("x") as output:
        return subprocess.run(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT).returncode


def main():
    status = {"task": "T072-AZ", "classification": "STARTED", "reference_reads": 0,
              "metrics": 0, "smoke": None, "full": []}
    save(RUNS / "status.json", status)
    try:
        smoke = RUNS / "smoke"
        result = command([sys.executable, str(EVIDENCE / "seal/launcher.py"),
                          "--runtime-root", str(RUNTIME), "--out", str(smoke)],
                         RUNTIME, RUNS / "smoke.log")
        status["smoke"] = json.loads((smoke / "receipt.json").read_text())
        if result or status["smoke"]["classification"] != "UHDLL_NATIVE_BASELINES_SMOKE_PASS":
            status["classification"] = status["smoke"]["classification"]
            save(RUNS / "status.json", status)
            return
        low = json.loads((EVIDENCE / "low_receipt.json").read_text())
        paths = [str(ROOT / "input/T072I_input_cache" / row["name"]) for row in low["files"]]
        bindings = json.loads((EVIDENCE / "seal/baseline_bindings.json").read_text())
        for method in METHODS:
            out = RUNS / "full" / method
            b = bindings[method]
            args = [sys.executable, "-m", "ttie." + ("retinex_exporter" if method == "retinexformer" else "snr_exporter"),
                    "--low", *paths, "--checkpoint", b["checkpoint"], "--config", b["config"],
                    "--out", str(out)]
            if method == "snr_aware":
                args += ["--source", b["source"]]
            status["full"].append({"method": method, "status": "STARTED", "run_count": 1})
            save(RUNS / "status.json", status)
            ret = command(args, RUNTIME, RUNS / (method + ".log"))
            status["full"][-1]["returncode"] = ret
            status["full"][-1]["status"] = "FAILED" if ret else "PROCESS_DONE"
            save(RUNS / "status.json", status)
            if ret:
                status["classification"] = "BLOCKED_FULL_" + method.upper()
                save(RUNS / "status.json", status)
                return
        status["classification"] = "FULL_PROCESSES_DONE_AWAITING_INDEPENDENT_VERIFY"
    except Exception as e:
        status.update(classification="BLOCKED_EXECUTION", error_type=type(e).__name__,
                      error=str(e), traceback=traceback.format_exc())
    save(RUNS / "status.json", status)


if __name__ == "__main__":
    main()

"""T075-B: add host-load context to freeze receipts (runtime receipts may be affected by other agents' CPU jobs).

usage: annotate_load.py DS ROW [NOTE]   Edits /root/autodl-tmp/TTIE/T075B/freeze/<DS>/<ROW>/freeze_receipt.json in place
(before it is copied anywhere). The window is [mtime of the smoke-check JSON, mtime of the full-run log].
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

TC = Path("/root/autodl-tmp/TTIE/T075B")
SMOKE = {"ours_step0": "ours_step0_smoke", "ours_ttt": "ours_ttt_smoke", "retinexformer": "retinexformer_smoke",
         "snr_aware": "snr_aware_smoke", "promptir": "promptir_smoke", "promptir_dctta": "dctta_smoke",
         "quadprior": "quadprior_smoke", "mr_illuminate": "mri_smoke"}
FULL = {"promptir_dctta": "dctta", "mr_illuminate": "mri"}
ds, row = sys.argv[1], sys.argv[2]
note = sys.argv[3] if len(sys.argv) > 3 else None
start = (TC / "runs" / ds / f"{SMOKE[row]}.smokecheck.json").stat().st_mtime
end = (TC / "logs" / ds / f"{FULL.get(row, row)}.log").stat().st_mtime
loads = []
for line in (TC / "logs" / "load_monitor.log").read_text().splitlines():
    parts = line.split()
    t = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    if start - 60 <= t <= end + 60:
        loads.append(float(parts[1]))
iso = lambda s: datetime.fromtimestamp(s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
path = TC / "freeze" / ds / row / "freeze_receipt.json"
receipt = json.loads(path.read_text())
receipt["host_load_context"] = {
    "full_run_window_utc": [iso(start), iso(end)],
    "loadavg_1min_samples": len(loads),
    "loadavg_1min_min_max": [min(loads), max(loads)] if loads else None,
    "monitor": "logs/load_monitor.log (60 s samples; started 2026-09-26T05:21:58Z)" if loads else "no samples (monitor not running)",
    "gpu_exclusive": "flock /root/autodl-tmp/TTIE/gpu.lock (GPU jobs serialized); CPU shared with other agents' jobs",
    **({"note": note} if note else {}),
}
path.write_text(json.dumps(receipt, indent=2) + "\n")
print(row, receipt["host_load_context"]["full_run_window_utc"], receipt["host_load_context"]["loadavg_1min_min_max"])

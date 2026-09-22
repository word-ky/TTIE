"""Poll only the A6000 GPU gate at five-minute intervals for T072-K.

The watcher never opens project inputs and exits at the first qualifying snapshot
or after the bounded 55-minute window.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


INTERVAL_SECONDS = 300
MAX_SNAPSHOTS = 12  # t=0 through t=55 minutes
FREE_MIN_MIB = 40960
PROCESS_MAX_MIB = 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def query(args: list[str]) -> tuple[str, list[str]]:
    proc = subprocess.run(
        ["nvidia-smi", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    raw = proc.stdout.strip()
    return raw, raw.splitlines() if raw else []


def parse_gpus(lines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in csv.reader(lines, skipinitialspace=True):
        if len(row) < 7:
            continue
        rows.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "memory_total_mib": int(row[3]),
                "memory_used_mib": int(row[4]),
                "memory_free_mib": int(row[5]),
                "utilization_gpu_percent": int(row[6]),
            }
        )
    return rows


def parse_apps(lines: list[str]) -> list[dict[str, object]]:
    if not lines or lines == ["No running processes found"]:
        return []
    apps: list[dict[str, object]] = []
    for row in csv.reader(lines, skipinitialspace=True):
        if len(row) < 4:
            continue
        apps.append(
            {
                "gpu_uuid": row[0],
                "pid": int(row[1]),
                "process_name": row[2],
                "used_memory_mib": int(row[3]),
            }
        )
    return apps


def snapshot(seq: int) -> dict[str, object]:
    gpu_args = [
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_args = [
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_raw, gpu_lines = query(gpu_args)
    app_raw, app_lines = query(app_args)
    gpus = parse_gpus(gpu_lines)
    apps = parse_apps(app_lines)
    apps_by_uuid: dict[str, list[dict[str, object]]] = {}
    for app in apps:
        apps_by_uuid.setdefault(str(app["gpu_uuid"]), []).append(app)
    qualifying: list[int] = []
    for gpu in gpus:
        device_apps = apps_by_uuid.get(str(gpu["uuid"]), [])
        if (
            gpu["name"] == "NVIDIA RTX A6000"
            and int(gpu["memory_free_mib"]) >= FREE_MIN_MIB
            and all(int(app["used_memory_mib"]) <= PROCESS_MAX_MIB for app in device_apps)
        ):
            qualifying.append(int(gpu["index"]))
    return {
        "seq": seq,
        "utc": utc_now(),
        "requirement": {
            "device": "NVIDIA RTX A6000",
            "minimum_free_mib": FREE_MIN_MIB,
            "maximum_unrelated_process_mib": PROCESS_MAX_MIB,
        },
        "gpu_query_raw": gpu_raw,
        "compute_apps_query_raw": app_raw,
        "gpus": gpus,
        "compute_apps": apps,
        "qualifying_indices": qualifying,
    }


def main() -> None:
    output = Path(__file__).with_name("watch.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    for seq in range(MAX_SNAPSHOTS):
        item = snapshot(seq)
        if item["qualifying_indices"]:
            item["event"] = "FIRST_QUALIFYING"
            output.open("a", encoding="utf-8").write(json.dumps(item, sort_keys=True) + "\n")
            return
        if seq == MAX_SNAPSHOTS - 1:
            item["event"] = "WINDOW_EXPIRED"
            output.open("a", encoding="utf-8").write(json.dumps(item, sort_keys=True) + "\n")
            return
        output.open("a", encoding="utf-8").write(json.dumps(item, sort_keys=True) + "\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

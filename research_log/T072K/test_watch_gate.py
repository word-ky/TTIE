from __future__ import annotations

from watch_gate import parse_apps, parse_gpus


def test_gate_parser_rejects_busy_a6000() -> None:
    gpus = parse_gpus(
        [
            "0, GPU-a, NVIDIA RTX A6000, 49140, 45000, 4140, 0",
            "1, GPU-b, NVIDIA RTX A6000, 49140, 1000, 48140, 0",
        ]
    )
    apps = parse_apps(["GPU-a, 123, VLLM::Worker_TP0, 44972"])
    assert gpus[0]["memory_free_mib"] < 40960
    assert apps[0]["used_memory_mib"] > 1024


def test_gate_parser_accepts_only_clean_a6000() -> None:
    gpus = parse_gpus(
        [
            "0, GPU-a, NVIDIA RTX A6000, 49140, 7190, 41950, 0",
            "1, GPU-b, NVIDIA RTX A6000, 49140, 1000, 48140, 0",
        ]
    )
    apps = parse_apps(["GPU-a, 123, helper, 1024", "GPU-b, 456, unrelated, 2048"])
    by_uuid = {str(app["gpu_uuid"]): app for app in apps}
    qualifying = [
        int(gpu["index"])
        for gpu in gpus
        if gpu["name"] == "NVIDIA RTX A6000"
        and int(gpu["memory_free_mib"]) >= 40960
        and all(
            int(app["used_memory_mib"]) <= 1024
            for app in apps
            if str(app["gpu_uuid"]) == str(gpu["uuid"])
        )
    ]
    assert qualifying == [0]

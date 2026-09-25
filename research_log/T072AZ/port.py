"""Mechanical T072-O host-path/device re-seal; no scientific asset edits."""
import argparse
import hashlib
import json
from pathlib import Path

OLD_SHARED = "/home/wenchang/asdasdsad/wjq/TTIE/shared"
OLD_INPUT = "/media/wenchang/F/wjq/TTIE/shared/t072i/uhdll/input"
OLD_DEVICE = "NVIDIA RTX A6000"
NEW_DEVICE = "NVIDIA GeForce RTX 4090"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    shared = a.root + "/shared"
    binds = json.loads((a.old / "baseline_bindings.json").read_text())
    for row in binds.values():
        for field in ("checkpoint", "source"):
            if field in row:
                row[field] = row[field].replace(OLD_SHARED, shared)
        row["files"] = {k.replace(OLD_SHARED, shared): v for k, v in row["files"].items()}
    dump(a.out / "baseline_bindings.json", binds)
    bind_sha = hashlib.sha256((a.out / "baseline_bindings.json").read_bytes()).hexdigest()
    spec = json.loads((a.old / "spec.json").read_text())
    spec["gate"]["name"] = NEW_DEVICE
    spec["smoke"]["path"] = spec["smoke"]["path"].replace(OLD_INPUT, a.root + "/input/T072I_input_cache")
    spec["sources"]["research_log/T071B/baseline_bindings.json"]["sha256"] = bind_sha
    dump(a.out / "spec.json", spec)
    old_sha = hashlib.sha256((a.old / "spec.json").read_bytes()).hexdigest()
    new_sha = hashlib.sha256((a.out / "spec.json").read_bytes()).hexdigest()
    launcher = (a.old / "launcher.py").read_text()
    assert launcher.count(old_sha) == 1 and launcher.count(OLD_DEVICE) == 1
    launcher = launcher.replace(old_sha, new_sha).replace(OLD_DEVICE, NEW_DEVICE)
    (a.out / "launcher.py").write_text(launcher)
    (a.out / "worker.py").write_bytes((a.old / "worker.py").read_bytes())
    dump(a.out / "port_manifest.json", {
        "task": "T072-AZ", "old_spec_sha256": old_sha, "new_spec_sha256": new_sha,
        "old_binding_sha256": hashlib.sha256((a.old / "baseline_bindings.json").read_bytes()).hexdigest(),
        "new_binding_sha256": bind_sha, "new_device": NEW_DEVICE,
        "old_shared": OLD_SHARED, "new_shared": shared,
        "old_input": OLD_INPUT, "new_input": a.root + "/input/T072I_input_cache",
        "worker_sha256": hashlib.sha256((a.out / "worker.py").read_bytes()).hexdigest(),
    })


if __name__ == "__main__":
    main()

"""Independent pre-inference audit of T072-AZ scientific binding and 150 lows."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for field in ("old", "new", "runtime", "root", "asset_manifest", "low_receipt"):
        p.add_argument("--" + field.replace("_", "-"), required=True)
    a = p.parse_args()
    old, new, runtime, root = map(Path, (a.old, a.new, a.runtime, a.root))
    port = json.loads((new / "port_manifest.json").read_text())
    assert sha(old / "spec.json") == port["old_spec_sha256"]
    assert sha(new / "spec.json") == port["new_spec_sha256"]
    assert sha(old / "baseline_bindings.json") == port["old_binding_sha256"]
    assert sha(new / "baseline_bindings.json") == port["new_binding_sha256"]
    assert sha(old / "worker.py") == sha(new / "worker.py") == port["worker_sha256"]
    orig_spec = json.loads((old / "spec.json").read_text())
    new_spec = json.loads((new / "spec.json").read_text())
    new_spec["gate"]["name"] = orig_spec["gate"]["name"]
    new_spec["smoke"]["path"] = orig_spec["smoke"]["path"]
    new_spec["sources"]["research_log/T071B/baseline_bindings.json"]["sha256"] = port["old_binding_sha256"]
    assert new_spec == orig_spec, "non-portability spec difference"
    assert new_spec["reference_reads"] == new_spec["metrics"] == 0
    normalized = (new / "baseline_bindings.json").read_text().replace(port["new_shared"], port["old_shared"])
    assert json.loads(normalized) == json.loads((old / "baseline_bindings.json").read_text()), "scientific binding difference"
    normalized_launcher = (new / "launcher.py").read_text().replace(port["new_spec_sha256"], port["old_spec_sha256"]).replace(port["new_device"], orig_spec["gate"]["name"])
    assert normalized_launcher == (old / "launcher.py").read_text(), "non-portability launcher difference"
    manifest = json.loads(Path(a.asset_manifest).read_text())
    old_runtime = manifest["runtime_root"]
    checked = 0
    for path, digest in manifest["files"].items():
        new_path = Path(path.replace(old_runtime, str(runtime)).replace(port["old_shared"], port["new_shared"]))
        if not new_path.is_absolute():
            new_path = runtime / new_path
        assert sha(new_path) == digest, new_path
        checked += 1
    low = json.loads(Path(a.low_receipt).read_text())
    assert low["count"] == len(low["files"]) == 150
    input_dir = root / "input/T072I_input_cache"
    assert {x.name for x in input_dir.iterdir()} == {row["name"] for row in low["files"]}
    for row in low["files"]:
        path = input_dir / row["name"]
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], path.name
        with Image.open(path) as im:
            assert im.mode == "RGB" and im.size == (3840, 2160), path.name
    print(json.dumps({"classification": "PASS_PREINFERENCE_PORT", "frozen_asset_files": checked,
                      "low_files": 150, "native_geometry": [2160, 3840, 3],
                      "reference_reads": 0, "metrics": 0}))


if __name__ == "__main__":
    main()

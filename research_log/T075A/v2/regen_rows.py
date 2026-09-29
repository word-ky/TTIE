"""Hash-committed, regenerate-at-metric-time storage for ``<row>_plus_D`` control rows (disk saving; orchestrator-
authorised 2026-09-26 for SID). ``ours_v2*`` rows always keep their files.

    strip   --outputs ROW   : the row must be frozen and every output file must match its manifest hashes; writes
                              <ROW>/regen_record.json (sha of manifest + per-image file/tensor hashes) and deletes the
                              output.pt.gz files (decision.json and output_manifest.json stay).
    restore --outputs ROW --source-dir SRC : regenerates every missing output.pt.gz from the frozen source row with
                              denoise_d.apply_d + denoise_d.save_tensor (deterministic gzip, mtime 0), and requires the
                              file SHA256 and tensor SHA256 to equal the frozen manifest bit for bit; fails closed.
Run ``restore`` before the reference gate (which re-hashes every output file) and before metrics; ``strip`` after.
Requires rows produced by a denoise_d.py whose save_tensor is deterministic (sha recorded as producer_sha256).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import denoise_d as dd  # noqa: E402


def strip(outputs):
    outputs = Path(outputs)
    manifest_path = outputs / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    dd.fail(manifest["method_id"].endswith("_plus_D"), "only <row>_plus_D control rows may be stripped")
    dd.fail(manifest.get("deterministic_gzip") is True, "row was not written with deterministic gzip; cannot regenerate")
    record = {"output_manifest_sha256": dd.sha256_file(manifest_path), "method_id": manifest["method_id"],
              "images": [{"low_name": r["low_name"], "output_file_sha256": r["output_file_sha256"],
                          "output_tensor_sha256": r["output_tensor_sha256"]} for r in manifest["rows"]]}
    for r in manifest["rows"]:
        path = outputs / Path(r["low_name"]).stem / "output.pt.gz"
        if path.exists():
            dd.fail(dd.sha256_file(path) == r["output_file_sha256"], f"file changed before strip: {path}")
    (outputs / "regen_record.json").write_text(json.dumps(record, indent=2))
    removed = 0
    for r in manifest["rows"]:
        path = outputs / Path(r["low_name"]).stem / "output.pt.gz"
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def restore(outputs, source_dir):
    outputs, source_dir = Path(outputs), Path(source_dir)
    manifest = json.loads((outputs / "output_manifest.json").read_bytes())
    source = json.loads((source_dir / "output_manifest.json").read_bytes())
    dd.fail(dd.sha256_file(source_dir / "output_manifest.json") == manifest["source_output_manifest_sha256"],
            "source manifest differs from the frozen binding")
    restored = 0
    for r, s in zip(manifest["rows"], source["rows"]):
        path = outputs / Path(r["low_name"]).stem / "output.pt.gz"
        if path.exists():
            dd.fail(dd.sha256_file(path) == r["output_file_sha256"], f"existing file differs: {path}")
            continue
        tensor, file_sha = dd.load_tensor(source_dir / Path(r["low_name"]).stem / "output.pt.gz")
        dd.fail(file_sha == s["output_file_sha256"] == r["source_output_file_sha256"], f"source file SHA {r['low_name']}")
        out, facts = dd.apply_d(tensor)
        dd.fail(dd.tensor_sha256(out) == r["output_tensor_sha256"] and facts["h"] == r["h"], f"regenerated tensor differs {r['low_name']}")
        dd.save_tensor(path, out)
        dd.fail(dd.sha256_file(path) == r["output_file_sha256"], f"regenerated file differs {r['low_name']}")
        restored += 1
    return restored


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["strip", "restore"])
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--source-dir", type=Path)
    a = p.parse_args()
    n = strip(a.outputs) if a.command == "strip" else restore(a.outputs, a.source_dir)
    print(json.dumps({"command": a.command, "row": str(a.outputs), "files": n}))


if __name__ == "__main__":
    main()

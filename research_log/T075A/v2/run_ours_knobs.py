"""GT-free producer: frozen T070-A Ours-TTT with the frozen SDSD-selected knob overrides (row ``ours_ttt_sdsd_knobs``).

Spec: research_log/T075A/v2_freeze.md. ``ours_v2_sdsd_knobs`` = D(this row) via denoise_d.py.
Run from the frozen Ours source root with the frozen process environment (CUBLAS_WORKSPACE_CONFIG and
*_NUM_THREADS unset), under flock on the shared GPU lock.

* No gate receipt, no GT, no metrics: ``ours_tuning`` (pinned T074-C file, unmodified) is imported only for its
  knob validation (``canonical_knobs``), its literal-substitution derivation from the hash-bound frozen source
  (``derive``) and its per-image runner (``run_group``: CLIP gate with the knob q_joint, derived trajectory,
  derived stop selection, no-active rule). Those are exactly the calls that produced the SDSD-indoor
  ``ours_ttt_target_tuned`` row.
* Knobs are fixed in this file (KNOBS); the command line cannot change them.
* Before the run, the default knobs are run on the first ``--repro-count`` images and must reproduce this target's
  frozen ``ours_ttt`` output tensors bit for bit (fails closed otherwise). ``--expect-manifest`` optionally requires
  every output tensor to equal a given row (used on SDSD-indoor against ``ours_ttt_target_tuned``).
* Rejection fallback (v2_freeze.md amendment, declared 2026-09-26 before any SMID GT): if the frozen code rejects an
  image under the knobs (AssertionError -> run_group status REJECTED), that image's output is a bitwise copy of this
  target's frozen default-knob ``ours_ttt`` output (file and tensor hashes checked against the frozen manifest), with
  decision_status = KNOBS_REJECTED_FALLBACK_DEFAULT; the count is reported in the manifest.
"""

import os
import sys

sys.path.insert(0, os.getcwd())  # frozen Ours source root

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

T074C = Path(os.environ.get("T074C_CODE", Path(__file__).resolve().parents[2] / "T074C"))
sys.path.insert(0, str(T074C))

METHOD_ID = "ours_ttt_sdsd_knobs"
KNOBS = {"q_joint": 0.35266535990213066, "exposure_target": 0.7}  # v2_freeze.md; T074-C selected setting 75f046d9...
SDSD_SETTING_ID = "75f046d9d6ab422f0c17d38ba091e66c4dab643b2884bfc038a3e282bbb77efd"
FALLBACK = "KNOBS_REJECTED_FALLBACK_DEFAULT"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def fail(condition, message):
    if not condition:
        raise RuntimeError(message)


def fallback_copy(frozen_row, frozen_dir, item, directory):
    """Declared rejection fallback: bitwise copy of the frozen default-knob ours_ttt output for this image."""
    import torch

    shape = [1, 3, item["height"], item["width"]]
    fail(frozen_row["low_name"] == item["name"] and frozen_row["low_sha256"] == item["sha256"]
         and frozen_row["shape"] == shape, f"frozen ours_ttt row mismatch at {item['name']}")
    source = Path(frozen_dir) / Path(item["name"]).stem / "output.pt.gz"
    fail(sha256_file(source) == frozen_row["output_file_sha256"], f"frozen ours_ttt file changed: {item['name']}")
    shutil.copyfile(source, Path(directory) / "output.pt.gz")
    fail(sha256_file(Path(directory) / "output.pt.gz") == frozen_row["output_file_sha256"], "fallback copy differs")
    with gzip.open(Path(directory) / "output.pt.gz", "rb") as stream:
        image = torch.load(stream, map_location="cpu", weights_only=True)
    fail(sha256_bytes(image.contiguous().numpy().tobytes()) == frozen_row["output_tensor_sha256"],
         f"frozen ours_ttt tensor changed: {item['name']}")
    return image, {"fallback_source": "frozen default-knob ours_ttt",
                   "fallback_source_decision_status": frozen_row["decision_status"],
                   "fallback_source_selected_step": frozen_row["decision"]["selected_step"]}


def save_tensor(path, tensor):
    import torch

    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)


def main():
    fail(__debug__, "run without -O: the frozen code's assert statements are part of the method")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="frozen T070-A execution manifest")
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--low-dir", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--frozen-ours-ttt", type=Path, required=True, help="this target's frozen ours_ttt row directory")
    p.add_argument("--repro-count", type=int, default=10)
    p.add_argument("--expect-manifest", type=Path, help="optional row whose output tensors must be reproduced")
    p.add_argument("--smoke-count", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    import torch
    import ours_tuning as OT
    from research_log.T070A.infer import FinalOurs
    from ttie.lolv2_gamma_core import native_rgb

    knobs = OT.canonical_knobs(KNOBS)
    fail(OT.setting_id(knobs) == SDSD_SETTING_ID, "knobs differ from the frozen SDSD-selected setting")
    defaults = OT.canonical_knobs({})
    fail(not a.out.exists(), f"{a.out} exists")
    files = json.loads(a.low_receipt.read_bytes())["files"]
    low_sha = sha256_file(a.low_receipt)
    fail(len(files) == a.expected_count, "low receipt count mismatch")
    frozen_path = a.frozen_ours_ttt / "output_manifest.json"
    frozen = json.loads(frozen_path.read_bytes())
    fail(frozen["low_receipt_sha256"] == low_sha and frozen["count"] == len(frozen["rows"]) == len(files),
         "frozen ours_ttt row bound to another low receipt / count")
    expect = json.loads(a.expect_manifest.read_bytes()) if a.expect_manifest else None

    model = FinalOurs(a.manifest)
    fail(frozen["execution_manifest_sha256"] == model.manifest_sha256, "frozen ours_ttt used another execution manifest")
    binding = model.manifest["source_binding"]
    variant = {"id": "knobs", "knobs": knobs, "derived": OT.derive(knobs, binding)[0]}
    default_variant = {"id": "default", "knobs": defaults, "derived": OT.derive(defaults, binding)[0]}

    def low(index):
        item = files[index]
        path = a.low_dir / item["name"]
        fail(sha256_file(path) == item["sha256"], f"low changed: {item['name']}")
        return native_rgb(path)

    # Frozen no-active abstentions copy Step0 bytes (a different code path), so only executed images are compared.
    executed = [i for i, r in enumerate(frozen["rows"]) if r["decision_status"] == "TTT_EXECUTED"]
    repro = executed[: a.repro_count]
    fail(len(repro) == min(a.repro_count, len(executed)) and (repro or a.repro_count == 0), "no executed image to compare")
    for index in repro:
        image, record = OT.run_group(model.scorer, model.gate, model.model, low(index), [default_variant], "cuda:0")["default"]
        fail(image is not None and sha256_bytes(image.contiguous().numpy().tobytes()) == frozen["rows"][index]["output_tensor_sha256"],
             f"default knobs do not reproduce frozen ours_ttt at {files[index]['name']}; stopping")
    print(f"default reproduction OK on {len(repro)} images", flush=True)
    repro = len(repro)

    partial = Path(str(a.out) + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    rows = []
    cohort = files[: a.smoke_count or None]
    for index, item in enumerate(cohort):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image, record = OT.run_group(model.scorer, model.gate, model.model, low(index), [variant], "cuda:0")["knobs"]
        shape = [1, 3, item["height"], item["width"]]
        directory = partial / Path(item["name"]).stem
        directory.mkdir()
        extra = {}
        if record["status"] == "REJECTED":
            fail(image is None, f"rejected image carries an output: {item['name']}")
            image, extra = fallback_copy(frozen["rows"][index], a.frozen_ours_ttt, item, directory)
            extra["rejection_error"] = record.get("error")
            record = {"status": FALLBACK, "selected_step": extra["fallback_source_selected_step"], "active": record["active"]}
        fail(image is not None and record["status"] in ("TTT_EXECUTED", "TTT_ABSTAIN_NO_ACTIVE_GATE", FALLBACK),
             f"unexpected record at {item['name']}: {record}")
        fail(list(image.shape) == shape and image.dtype == torch.float32 and bool(torch.isfinite(image).all()),
             f"output geometry at {item['name']}")
        digest = sha256_bytes(image.contiguous().numpy().tobytes())
        if expect is not None:
            fail(digest == expect["rows"][index]["output_tensor_sha256"], f"expected row not reproduced at {item['name']}")
        if record["status"] != FALLBACK:
            save_tensor(directory / "output.pt.gz", image)
        row = {"method": "Ours-TTT (SDSD-selected knobs)", "low_name": item["name"], "low_sha256": item["sha256"],
               "execution_manifest_sha256": model.manifest_sha256, "setting_id": SDSD_SETTING_ID,
               "decision_status": record["status"], "selected_step": record["selected_step"], "active": record["active"],
               **({"decision": record["decision"]} if "decision" in record else {}), **extra,
               "shape": shape, "dtype": "torch.float32", "output_tensor_sha256": digest,
               "output_file_sha256": sha256_file(directory / "output.pt.gz"),
               "whole_run_seconds": time.perf_counter() - started,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved()}
        (directory / "decision.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"{index + 1}/{len(cohort)} {item['name']} {record['status']} step={record['selected_step']}", flush=True)
    manifest = {"method": "Ours-TTT (SDSD-selected knobs)", "method_id": METHOD_ID, "count": len(rows),
                "low_receipt_sha256": low_sha, "execution_manifest_sha256": model.manifest_sha256,
                "knobs": knobs, "setting_id": SDSD_SETTING_ID, "knob_overrides": KNOBS,
                "frozen_ours_ttt_manifest_sha256": sha256_file(frozen_path), "default_reproduction_images": repro,
                "expected_manifest_sha256": sha256_file(a.expect_manifest) if a.expect_manifest else None,
                "no_active_abstentions": sum(r["decision_status"] == "TTT_ABSTAIN_NO_ACTIVE_GATE" for r in rows),
                "knobs_rejected_fallback_count": sum(r["decision_status"] == FALLBACK for r in rows),
                "fallback_rule": "v2_freeze.md amendment: rejected image -> bitwise copy of frozen default-knob ours_ttt output",
                "producer": "research_log/T075A/v2/run_ours_knobs.py", "producer_sha256": sha256_file(__file__),
                "ours_tuning_sha256": sha256_file(OT.__file__), "promotable": not a.smoke_count,
                "reference_reads": 0, "metrics": 0, "rows": rows}
    with open(partial / "output_manifest.json", "x") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(partial, a.out)
    print(f"OUTPUT_MANIFEST_SHA256 {sha256_file(a.out / 'output_manifest.json')}", flush=True)


if __name__ == "__main__":
    main()

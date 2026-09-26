"""Independent verifier for the T075-A v2 rows (no reference access, no metrics).

Kinds:
  plus_d  ``ours_v2``, ``ours_v2_sdsd_knobs`` and ``<row>_plus_D``: schema, low binding, per-image file/tensor hashes,
          decision.json == manifest row, binding to the source row's manifest and per-image source hashes, 8-bit
          quantisation of every output value, h == 6 * 255 * sigma_hat, and an independent re-execution of D
          (own Immerkaer implementation by array slicing, tolerance 1e-9 relative on sigma_hat; then OpenCV NLM with
          the recorded h must give the stored tensor bit for bit) on every ``--recompute-stride``-th image.
  knobs   ``ours_ttt_sdsd_knobs``: schema, hashes, fixed knobs / setting id, execution-manifest binding to the frozen
          ours_ttt row, decision fields and abstention consistency.
Does not import the producers.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

KNOBS_SETTING_ID = "75f046d9d6ab422f0c17d38ba091e66c4dab643b2884bfc038a3e282bbb77efd"
KNOB_OVERRIDES = {"q_joint": 0.35266535990213066, "exposure_target": 0.7}
KAPPA = 6.0


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    import torch

    raw = Path(path).read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        return torch.load(stream, map_location="cpu", weights_only=True), sha256_bytes(raw)


def sigma_by_slicing(x):
    """Immerkaer sigma, interior pixels, written out as the 3x3 stencil; RGB mean."""
    out = []
    for c in range(3):
        v = x[..., c]
        r = (v[:-2, :-2] - 2 * v[:-2, 1:-1] + v[:-2, 2:] - 2 * v[1:-1, :-2] + 4 * v[1:-1, 1:-1]
             - 2 * v[1:-1, 2:] + v[2:, :-2] - 2 * v[2:, 1:-1] + v[2:, 2:])
        out.append(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)
    return float(np.mean(out))


def verify_image(row, item, root, recompute, source_dir, source_row):
    import cv2
    import torch

    shape = [1, 3, item["height"], item["width"]]
    check(row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"], f"low binding {item['name']}")
    check(row["shape"] == shape and row["dtype"] == "torch.float32", f"shape/dtype {item['name']}")
    d = Path(root) / Path(item["name"]).stem
    check(json.loads((d / "decision.json").read_bytes()) == row, f"decision.json differs {item['name']}")
    tensor, file_sha = load(d / "output.pt.gz")
    check(file_sha == row["output_file_sha256"], f"output file SHA {item['name']}")
    check(list(tensor.shape) == shape and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
          f"tensor geometry {item['name']}")
    check(sha256_bytes(tensor.contiguous().numpy().tobytes()) == row["output_tensor_sha256"], f"tensor SHA {item['name']}")
    if source_row is None:
        return (d / "output.pt.gz").stat().st_size, False
    array = tensor.numpy()
    levels = np.rint(array * 255)
    check(bool((levels >= 0).all() and (levels <= 255).all()), f"output outside [0,1] {item['name']}")
    check(np.array_equal(levels.astype(np.float32) / np.float32(255), array), f"output not 8-bit quantised {item['name']}")
    check(row["source_output_tensor_sha256"] == source_row["output_tensor_sha256"]
          and row["source_output_file_sha256"] == source_row["output_file_sha256"], f"source binding {item['name']}")
    check(row["h"] == KAPPA * 255.0 * row["sigma_hat"] and row["sigma_hat"] >= 0, f"h rule {item['name']}")
    if not recompute:
        return (d / "output.pt.gz").stat().st_size, False
    source, source_file_sha = load(Path(source_dir) / Path(item["name"]).stem / "output.pt.gz")
    check(source_file_sha == source_row["output_file_sha256"], f"source file SHA {item['name']}")
    check(sha256_bytes(source.contiguous().numpy().tobytes()) == source_row["output_tensor_sha256"], f"source tensor {item['name']}")
    x = np.clip(source[0].numpy().transpose(1, 2, 0).astype(np.float64), 0, 1)
    sigma = sigma_by_slicing(x)
    check(abs(sigma - row["sigma_hat"]) <= 1e-9 * max(sigma, 1e-12), f"sigma_hat mismatch {item['name']}: {sigma} vs {row['sigma_hat']}")
    cv2.setNumThreads(1)
    u8 = np.ascontiguousarray((np.round(x * 255)).astype(np.uint8)[:, :, ::-1])
    den = cv2.fastNlMeansDenoisingColored(u8, None, row["h"], row["h"], 7, 21)[:, :, ::-1]
    again = den.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255)
    check(np.array_equal(again, array), f"D re-execution differs {item['name']}")
    return (d / "output.pt.gz").stat().st_size, True


def verify(kind, method_id, low_receipt, outputs, expected_count, source_dir=None, source_row_id=None,
           frozen_ours_ttt=None, recompute_stride=1):
    files = json.loads(Path(low_receipt).read_bytes())["files"]
    manifest_path = Path(outputs) / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    names = [f["name"] for f in files]
    check(len(names) == len(set(names)) == expected_count, "low receipt count")
    check(manifest["method_id"] == method_id, "method_id")
    check(manifest["count"] == len(manifest["rows"]) == expected_count, "manifest count")
    check(manifest["low_receipt_sha256"] == sha256_file(low_receipt), "low receipt binding")
    check(manifest["reference_reads"] == 0 and manifest["metrics"] == 0, "reference/metric access recorded")
    check(manifest.get("promotable") is True, "not promotable (smoke)")
    extra = {}
    source = None
    if kind == "plus_d":
        source_path = Path(source_dir) / "output_manifest.json"
        source = json.loads(source_path.read_bytes())
        check(manifest["source_row"] == source_row_id, "source row id")
        check(manifest["source_output_manifest_sha256"] == sha256_file(source_path), "source manifest binding")
        check(source["count"] == expected_count and source["low_receipt_sha256"] == manifest["low_receipt_sha256"], "source count/low")
        spec = manifest["d_spec"]
        check(spec["kappa"] == KAPPA and spec["template_window"] == 7 and spec["search_window"] == 21, "D spec")
        extra = {"source_row": source_row_id, "source_output_manifest_sha256": manifest["source_output_manifest_sha256"],
                 "recompute_stride": recompute_stride}
    else:
        check(kind == "knobs", f"unknown kind {kind}")
        frozen_path = Path(frozen_ours_ttt) / "output_manifest.json"
        frozen = json.loads(frozen_path.read_bytes())
        check(manifest["setting_id"] == KNOBS_SETTING_ID and manifest["knob_overrides"] == KNOB_OVERRIDES, "knobs")
        check(manifest["knobs"]["q_joint"] == KNOB_OVERRIDES["q_joint"]
              and manifest["knobs"]["exposure_target"] == KNOB_OVERRIDES["exposure_target"], "knobs values")
        check(manifest["frozen_ours_ttt_manifest_sha256"] == sha256_file(frozen_path), "frozen ours_ttt binding")
        check(manifest["execution_manifest_sha256"] == frozen["execution_manifest_sha256"], "execution manifest")
        check(manifest["default_reproduction_images"] >= 1, "no default reproduction recorded")
    total, recomputed, abstained = 0, 0, 0
    for index, (item, row) in enumerate(zip(files, manifest["rows"])):
        if kind == "knobs":
            check(row["execution_manifest_sha256"] == manifest["execution_manifest_sha256"] and row["setting_id"] == KNOBS_SETTING_ID,
                  f"row binding {item['name']}")
            check(len(row["active"]) == 4 and all(isinstance(v, bool) for v in row["active"]), "active")
            if row["decision_status"] == "TTT_EXECUTED":
                check(0 <= row["selected_step"] <= 27 and any(row["active"]), f"executed row {item['name']}")
            else:
                check(row["decision_status"] == "TTT_ABSTAIN_NO_ACTIVE_GATE" and row["selected_step"] == 0
                      and not any(row["active"]), f"abstention row {item['name']}")
                abstained += 1
        size, did = verify_image(row, item, outputs, kind == "plus_d" and index % recompute_stride == 0,
                                 source_dir, source["rows"][index] if source else None)
        total += size
        recomputed += did
    if kind == "knobs":
        check(manifest["no_active_abstentions"] == abstained, "abstention count")
        extra = {"no_active_abstentions": abstained, "ttt_executed_count": expected_count - abstained}
    else:
        extra["recomputed_images"] = recomputed
    return {"classification": f"T075A_{kind.upper()}_OUTPUTS_VERIFIED", "method_id": method_id, "count": expected_count,
            "output_manifest_sha256": sha256_file(manifest_path), "low_receipt_sha256": manifest["low_receipt_sha256"],
            "compressed_output_bytes": total, **extra, "verifier_sha256": sha256_file(__file__),
            "reference_reads": 0, "metrics": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--kind", choices=["plus_d", "knobs"], required=True)
    p.add_argument("--method-id", required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--source-row")
    p.add_argument("--frozen-ours-ttt", type=Path)
    p.add_argument("--recompute-stride", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.kind, a.method_id, a.low_receipt, a.outputs, a.expected_count, a.source_dir, a.source_row,
                    a.frozen_ours_ttt, a.recompute_stride)
    with open(a.out, "x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Ours v2 denoiser D and the GT-free ``<row> + D`` producer (spec: research_log/T075A/v2_freeze.md).

D(image) for one frozen output tensor [1, 3, H, W] float32:
  1. x = clip(float64(image), 0, 1), HWC RGB.
  2. sigma_hat = Immerkaer (1996) noise estimate of x: per channel, sqrt(pi/2) * mean(|x (*) M|) / 6 over the
     interior (H-2)x(W-2) pixels, M = [[1,-2,1],[-2,4,-2],[1,-2,1]] (scipy.ndimage.convolve); mean over R, G, B.
  3. h = hColor = KAPPA * 255 * sigma_hat, KAPPA = 6 (frozen).
  4. Quantisation to 8 bit (the NLM implementation is uint8-only): u8 = round(x * 255) (numpy round-half-even)
     as uint8, channel order BGR.
  5. cv2.fastNlMeansDenoisingColored(u8, None, h, h, 7, 21), single OpenCV thread.
  6. Back to RGB float32: out = float32(u8_denoised) / float32(255), i.e. every output value is k/255.
The only information used is the image itself (no GT, no other image, no training).

Producer: reads a frozen source row (output_manifest.json + <stem>/output.pt.gz), checks every file and tensor
hash, applies D, and writes a new row in the lossless T074-B schema ([1,3,H,W] float32 output.pt.gz +
decision.json per image, output_manifest.json with reference_reads = metrics = 0). Writes to <out>.partial and
renames at the end. CPU only.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

KAPPA = 6.0
TEMPLATE_WINDOW = 7
SEARCH_WINDOW = 21
IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
SPEC = {
    "denoiser": "cv2.fastNlMeansDenoisingColored(u8_bgr, None, h, h, 7, 21)",
    "h_rule": "h = hColor = kappa * 255 * sigma_hat; sigma_hat = Immerkaer estimate of clip(float64 output, 0, 1)",
    "kappa": KAPPA,
    "template_window": TEMPLATE_WINDOW,
    "search_window": SEARCH_WINDOW,
    "quantisation": "u8 = uint8(np.round(clip(x,0,1) * 255)) before D; output = float32(u8_denoised) / float32(255)",
    "spec_document": "research_log/T075A/v2_freeze.md",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(condition, message):
    if not condition:
        raise RuntimeError(message)


def immerkaer_sigma(x):
    """x: HWC float64 in [0, 1]. Mean over channels of Immerkaer's sigma (interior pixels only)."""
    from scipy.ndimage import convolve

    values = []
    for channel in range(3):
        response = convolve(x[..., channel], IMMERKAER_KERNEL, mode="reflect")[1:-1, 1:-1]
        values.append(np.sqrt(np.pi / 2) * np.abs(response).mean() / 6.0)
    return float(np.mean(values))


def apply_d(tensor):
    """tensor: torch float32 [1, 3, H, W]. Returns (torch float32 [1, 3, H, W], {"sigma_hat", "h"})."""
    import cv2
    import torch

    cv2.setNumThreads(1)
    fail(tensor.dtype == torch.float32 and tensor.ndim == 4 and tensor.shape[:2] == (1, 3), "D expects [1,3,H,W] float32")
    fail(bool(torch.isfinite(tensor).all()), "nonfinite input to D")
    x = np.clip(tensor[0].permute(1, 2, 0).numpy().astype(np.float64), 0.0, 1.0)
    sigma = immerkaer_sigma(x)
    h = KAPPA * 255.0 * sigma
    bgr = np.ascontiguousarray(np.round(x[..., ::-1] * 255.0).astype(np.uint8))
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, TEMPLATE_WINDOW, SEARCH_WINDOW)
    rgb = np.ascontiguousarray(denoised[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / np.float32(255)
    return torch.from_numpy(np.ascontiguousarray(rgb))[None], {"sigma_hat": sigma, "h": h}


def load_tensor(path):
    import torch

    raw = Path(path).read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        return torch.load(stream, map_location="cpu", weights_only=True), sha256_bytes(raw)


def save_tensor(path, tensor):
    import torch

    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)


def tensor_sha256(tensor):
    return sha256_bytes(tensor.contiguous().numpy().tobytes())


def _work(task):
    import torch

    torch.set_num_threads(1)
    source_dir, out_dir, item, source_row, method = task
    started = time.perf_counter()
    stem = Path(item["name"]).stem
    tensor, file_sha = load_tensor(Path(source_dir) / stem / "output.pt.gz")
    shape = [1, 3, item["height"], item["width"]]
    fail(file_sha == source_row["output_file_sha256"], f"source output file SHA mismatch: {item['name']}")
    fail(list(tensor.shape) == shape and tensor.dtype == torch.float32, f"source geometry/dtype mismatch: {item['name']}")
    fail(tensor_sha256(tensor) == source_row["output_tensor_sha256"], f"source tensor SHA mismatch: {item['name']}")
    output, facts = apply_d(tensor)
    fail(list(output.shape) == shape and bool(torch.isfinite(output).all()), "D output geometry")
    directory = Path(out_dir) / stem
    directory.mkdir(parents=True)
    save_tensor(directory / "output.pt.gz", output)
    row = {"method": method, "low_name": item["name"], "low_sha256": item["sha256"],
           "source_output_tensor_sha256": source_row["output_tensor_sha256"],
           "source_output_file_sha256": source_row["output_file_sha256"],
           **({"source_decision_status": source_row["decision_status"]} if "decision_status" in source_row else {}),
           "sigma_hat": facts["sigma_hat"], "h": facts["h"], "shape": shape, "dtype": "torch.float32",
           "output_tensor_sha256": tensor_sha256(output), "output_file_sha256": sha256_file(directory / "output.pt.gz"),
           "whole_run_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": 0}
    (directory / "decision.json").write_text(json.dumps(row, indent=2))
    return row


def produce(source_dir, low_receipt, expected_count, method_id, source_row_id, out, workers=4,
            source_freeze=None, smoke_count=0):
    import cv2

    source_dir, out = Path(source_dir), Path(out)
    fail(not out.exists(), f"{out} exists")
    files = json.loads(Path(low_receipt).read_bytes())["files"]
    low_sha = sha256_file(low_receipt)
    manifest_path = source_dir / "output_manifest.json"
    source_manifest_sha = sha256_file(manifest_path)
    source = json.loads(manifest_path.read_bytes())
    fail(len(files) == expected_count == source["count"] == len(source["rows"]), "count mismatch")
    fail(source["low_receipt_sha256"] == low_sha, "source row bound to another low receipt")
    fail(source["reference_reads"] == 0 and source["metrics"] == 0, "source row records reference/metric access")
    fail(source.get("promotable", True) is True, "source manifest is not promotable")
    freeze_sha = None
    if source_freeze is not None:
        freeze = json.loads(Path(source_freeze).read_bytes())
        freeze_sha = sha256_file(source_freeze)
        fail(freeze["classification"] == "FROZEN_OUTPUTS" and freeze["method_id"] == source_row_id,
             "source freeze receipt identity mismatch")
        fail(freeze["output_manifest_sha256"] == source_manifest_sha, "source manifest differs from its freeze receipt")
    for item, row in zip(files, source["rows"]):
        fail(row["low_name"] == item["name"] and row["low_sha256"] == item["sha256"], f"source row/low mismatch {item['name']}")
    method = "Ours-v2 (Ours-TTT + D)" if method_id.startswith("ours_v2") else f"{source_row_id} + D"
    partial = Path(str(out) + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    cohort = list(zip(files, source["rows"]))[: smoke_count or None]
    tasks = [(str(source_dir), str(partial), item, row, method) for item, row in cohort]
    with get_context("spawn").Pool(workers) as pool:
        rows = []
        for index, row in enumerate(pool.imap(_work, tasks, chunksize=1)):
            rows.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(tasks):
                print(f"{method_id}: {index + 1}/{len(tasks)}", flush=True)
    manifest = {"method": method, "method_id": method_id, "count": len(rows), "low_receipt_sha256": low_sha,
                "source_row": source_row_id, "source_output_manifest_sha256": source_manifest_sha,
                "source_freeze_receipt_sha256": freeze_sha, "d_spec": SPEC,
                "producer": "research_log/T075A/v2/denoise_d.py", "producer_sha256": sha256_file(__file__),
                "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "cv2": cv2.__version__},
                "workers": workers, "promotable": not smoke_count, "reference_reads": 0, "metrics": 0, "rows": rows}
    with open(partial / "output_manifest.json", "x") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(partial, out)
    digest = sha256_file(out / "output_manifest.json")
    print(f"OUTPUT_MANIFEST_SHA256 {digest}", flush=True)
    return digest


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source-dir", type=Path, required=True, help="frozen source row directory (output_manifest.json)")
    p.add_argument("--source-row", required=True, help="source row id, e.g. retinexformer or ours_ttt")
    p.add_argument("--source-freeze", type=Path, help="source row freeze_receipt.json (binds its manifest SHA)")
    p.add_argument("--method-id", required=True, help="ours_v2 | ours_v2_sdsd_knobs | <row>_plus_D")
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke-count", type=int, default=0, help="first N images only; manifest not promotable")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    fail(a.workers <= 6, "at most 6 workers (shared 12-CPU host)")
    produce(a.source_dir, a.low_receipt, a.expected_count, a.method_id, a.source_row, a.out, a.workers,
            a.source_freeze, a.smoke_count)


if __name__ == "__main__":
    main()

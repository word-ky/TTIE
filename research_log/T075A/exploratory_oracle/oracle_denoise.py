"""EXPLORATORY_ORACLE_NOT_FOR_PAPER -- T075-A task 2(c): Ours output + classical denoiser, strength by GT.

Inputs per image: frozen ours_ttt output, target-tuned output (both hash-checked via metrics.load_output),
and the PSNR-oracle-step states from oracle_steps.py (default / tuned). Denoisers: OpenCV
fastNlMeansDenoisingColored (uint8, h = hColor on a fixed grid). CBM3D was dropped: bm3d 4.0.3 spawns
one thread per core and needs ~35 s/image at 960x512 on the shared host (too slow for a strength grid).
sigma_hat = Immerkaer estimate of the input is recorded for a blind-strength analysis. Also a counterfactual: SSIM/PSNR of GT + white Gaussian
noise. Metrics: frozen T071-A via metrics.score. GT only via metrics.Context.reference. CPU only.
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

LABEL = "EXPLORATORY_ORACLE_NOT_FOR_PAPER"
K = Path(os.environ.get("T074C_CODE", "/root/autodl-tmp/TTIE/T074C/code/research_log/T074C"))
sys.path.insert(0, str(K))
import metrics as M  # noqa: E402

NLM_H = (2, 4, 6, 8, 10, 12, 15, 18, 22, 27)
GT_NOISE = (0.005, 0.01, 0.0126, 0.015, 0.02)
REFS = CORE = None
IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def immerkaer(img):
    from scipy.ndimage import convolve
    return float(np.mean([np.sqrt(np.pi / 2) * np.abs(convolve(img[..., c], IMM, mode="reflect")[1:-1, 1:-1]).mean() / 6
                          for c in range(3)]))


def work(task):
    import cv2
    cv2.setNumThreads(1)
    index, name, loader = task
    y = REFS[index]
    if name == "gt_plus_noise":
        rng = np.random.Generator(np.random.PCG64(1000 + index))
        n = rng.standard_normal(y.shape)
        return {"index": index, "input": name,
                "noise": {str(s): M.score(np.asarray(y, np.float64) + s * n, y, CORE) for s in GT_NOISE}}
    kind, arg = loader
    if kind == "row":
        path, row, item = arg
        x = M.load_output(path, row, item)
    else:
        x = np.load(arg)
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    out = {"index": index, "input": name, "base": M.score(x, y, CORE), "sigma_hat": immerkaer(x), "nlm": {}}
    bgr = np.round(x[..., ::-1] * 255).astype(np.uint8)
    for h in NLM_H:
        d = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(bgr), None, h, h, 7, 21)
        out["nlm"][str(h)] = M.score(d[..., ::-1].astype(np.float64) / 255.0, y, CORE)
    return out


def main():
    global REFS, CORE
    p = argparse.ArgumentParser()
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=96)
    a = p.parse_args()
    ctx = M.Context("SDSD_indoor", a.low_receipt, a.opaque_manifest, a.gate_receipt)
    CORE = ctx.core
    sources = M.row_sources(ctx, a.tuned_row)
    REFS = [ctx.reference(i) for i in range(len(ctx.files))]
    tasks = []
    for i, item in enumerate(ctx.files):
        stem = Path(item["name"]).stem
        for row_id in ("ours_ttt", M.TUNED):
            manifest, _, root, _ = sources[row_id]
            tasks.append((i, row_id, ("row", (root / stem / "output.pt.gz", manifest["rows"][i], item))))
        for setting in ("default", "tuned"):
            tasks.append((i, f"{setting}_psnr_oracle_step", ("npy", a.oracle_dir / "images" / setting / f"{stem}__psnr_oracle.npy")))
        tasks.append((i, "gt_plus_noise", None))
    with get_context("fork").Pool(a.workers) as pool:
        recs = pool.map(work, tasks, chunksize=1)
    payload = {"label": LABEL, "task": "T075A-oracle-denoise", "gate_receipt_sha256": ctx.receipt_sha256,
               "oracle_denoise.py_sha256": M.sha256_file(__file__), "metrics.py_sha256": M.sha256_file(M.__file__),
               "nlm": "cv2.fastNlMeansDenoisingColored(uint8 BGR, h=hColor, template 7, search 21)",
               "nlm_h": NLM_H,
               "gt_noise_sigma": GT_NOISE, "records": recs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(payload, f, allow_nan=False)
    with open(Path(str(a.out) + ".reads.json"), "x") as f:
        json.dump(ctx.reads, f)
    print("DONE", len(recs), flush=True)


if __name__ == "__main__":
    main()

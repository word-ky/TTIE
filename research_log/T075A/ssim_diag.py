"""T075-A task 1: SSIM diagnosis on SDSD-indoor from existing frozen outputs (analysis only, CPU).

GT is read only through ``metrics.Context.reference`` (gate-validated official loader path). Outputs are
read through ``metrics.load_output`` (file + tensor hash checks). Metrics are the frozen T071-A functions
(``ctx.core.metrics``); the SSIM decomposition re-uses the frozen filter settings (gaussian sigma 1.5,
truncate 3.5, reflect, C1=.01^2, C2=.03^2, population covariance, RGB mean).
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, convolve, sobel

K = Path(os.environ.get("T074C_CODE", "/root/autodl-tmp/TTIE/T074C/code/research_log/T074C"))
sys.path.insert(0, str(K))
import metrics as M  # noqa: E402

C1, C2 = 0.01 ** 2, 0.03 ** 2
REFS = None
CORE = None


def filt(v):
    return gaussian_filter(v, sigma=(1.5, 1.5, 0), truncate=3.5, mode="reflect")


def ssim_parts(x, y):
    """Frozen rgb_ssim maps split: l, cs (=c*s up to C3), c, s. Means over pixels and channels."""
    ux, uy = filt(x), filt(y)
    vx = np.maximum(filt(x * x) - ux * ux, 0)
    vy = np.maximum(filt(y * y) - uy * uy, 0)
    vxy = filt(x * y) - ux * uy
    l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
    cs = (2 * vxy + C2) / (vx + vy + C2)
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    c = (2 * sx * sy + C2) / (vx + vy + C2)
    s = (vxy + C2 / 2) / (sx * sy + C2 / 2)
    return {"ssim": float((l * cs).mean()), "l": float(l.mean()), "cs": float(cs.mean()),
            "c": float(c.mean()), "s": float(s.mean())}


IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def immerkaer(img):
    """Immerkaer (1996) fast noise sigma per channel, averaged over RGB, [0,1] units."""
    out = []
    for ch in range(3):
        r = convolve(img[..., ch], IMM, mode="reflect")[1:-1, 1:-1]
        out.append(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)
    return float(np.mean(out))


def flat_mask(y):
    """GT-flat pixels: lowest 30% of blurred-GT gradient magnitude (gray)."""
    g = gaussian_filter(y.mean(axis=2), 2.0)
    mag = np.hypot(sobel(g, 0), sobel(g, 1))
    return mag <= np.quantile(mag, 0.30)


def hp_sigma(img, mask):
    """Robust sigma (1.4826 MAD) of the high-pass residual img - gauss(img, 1.0) inside mask, RGB mean."""
    r = img - gaussian_filter(img, sigma=(1.0, 1.0, 0), truncate=3.5, mode="reflect")
    vals = []
    for ch in range(3):
        v = r[..., ch][mask]
        vals.append(1.4826 * np.median(np.abs(v - np.median(v))))
    return float(np.mean(vals))


def work(task):
    index, row_id, path, row, item = task
    os.environ["OMP_NUM_THREADS"] = "1"
    y = np.ascontiguousarray(REFS[index], dtype=np.float64)
    x = np.clip(np.ascontiguousarray(M.load_output(path, row, item), dtype=np.float64), 0, 1)
    base = CORE.metrics(x, y)
    scale = y.mean() / max(x.mean(), 1e-8)
    xb = np.ascontiguousarray(np.clip(x * scale, 0, 1))
    cscale = y.mean(axis=(0, 1)) / np.maximum(x.mean(axis=(0, 1)), 1e-8)
    xc = np.ascontiguousarray(np.clip(x * cscale, 0, 1))
    mb, mc = CORE.metrics(xb, y), CORE.metrics(xc, y)
    mask = flat_mask(y)
    rec = {"index": index, "row": row_id, "name": item["name"], "cluster": item["cluster"],
           "psnr": base["psnr"], "ssim": base["rgb_ssim"],
           "bm_psnr": mb["psnr"], "bm_ssim": mb["rgb_ssim"], "bm_scale": float(scale),
           "cm_psnr": mc["psnr"], "cm_ssim": mc["rgb_ssim"],
           "mean_out": float(x.mean()), "mean_gt": float(y.mean()),
           "imm_sigma_out": immerkaer(x), "imm_sigma_gt": immerkaer(y),
           "hp_sigma_out_flat": hp_sigma(x, mask), "hp_sigma_bm_flat": hp_sigma(xb, mask),
           "hp_sigma_gt_flat": hp_sigma(y, mask),
           "sat_hi_frac": float((x >= 1).mean()), "sat_hi_frac_bm": float((xb >= 1).mean())}
    rec.update({f"raw_{k}": v for k, v in ssim_parts(x, y).items()})
    rec.update({f"bm_{k}": v for k, v in ssim_parts(xb, y).items() if k != "ssim"})
    return rec


def main():
    global REFS, CORE
    p = argparse.ArgumentParser()
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=48)
    a = p.parse_args()
    ctx = M.Context("SDSD_indoor", a.low_receipt, a.opaque_manifest, a.gate_receipt)
    CORE = ctx.core
    sources = M.row_sources(ctx, a.tuned_row)
    REFS = [ctx.reference(i) for i in range(len(ctx.files))]
    tasks = []
    for row_id, (manifest, _, root, _) in sources.items():
        for i, item in enumerate(ctx.files):
            tasks.append((i, row_id, root / Path(item["name"]).stem / "output.pt.gz", manifest["rows"][i], item))
    with get_context("fork").Pool(a.workers) as pool:
        recs = pool.map(work, tasks, chunksize=4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": "T075A-ssim-diagnosis", "gate_receipt_sha256": ctx.receipt_sha256,
               "reference_reads": len(ctx.reads), "ssim_diag.py_sha256": M.sha256_file(__file__),
               "metrics.py_sha256": M.sha256_file(M.__file__), "records": recs}
    with open(a.out, "x") as f:
        json.dump(payload, f, allow_nan=False)
    with open(Path(str(a.out) + ".reads.json"), "x") as f:
        json.dump(ctx.reads, f)
    print("DONE", len(recs), flush=True)


if __name__ == "__main__":
    main()

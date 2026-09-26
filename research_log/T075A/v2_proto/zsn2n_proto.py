"""T075-A task 3 prototype (DEVELOPMENT-ON-TEST, SDSD-indoor): Ours output + per-image ZS-N2N denoiser.

ZS-N2N (Mansour & Heckel, CVPR 2023): a 2x(3x3 conv, 48ch, LeakyReLU .2) + 1x1 conv residual network trained
from scratch on the single image to be denoised, with the pair-downsampler residual + consistency loss;
Adam lr 1e-3, 2000 iterations, lr x0.5 at 1500. Only the current Ours output is used (no GT, no other image).
Phase A (GPU, GT-free): denoise every image of the given rows, save float32 HWC npy + timing/VRAM.
Phase B (CPU): score with the frozen T071-A metric; GT only via metrics.Context.reference.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

K = Path(os.environ.get("T074C_CODE", "/root/autodl-tmp/TTIE/T074C/code/research_log/T074C"))
sys.path.insert(0, str(K))
import metrics as M  # noqa: E402

LABEL = "DEVELOPMENT_ON_TEST_SDSD_INDOOR"
REFS = CORE = None


def denoise(img_hwc, iters, seed, device, pd=1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    x = torch.from_numpy(np.ascontiguousarray(img_hwc.transpose(2, 0, 1)))[None].float().to(device)
    H, W = x.shape[-2:]
    if pd > 1:  # pixel-shuffle downsampling: pd*pd sub-images as one batch (decorrelates neighbouring noise)
        x = F.pixel_unshuffle(x, pd).reshape(1, 3, pd * pd, H // pd, W // pd).permute(2, 1, 0, 3, 4).reshape(pd * pd, 3, H // pd, W // pd)
    net = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv2d(48, 48, 3, padding=1),
                        nn.LeakyReLU(0.2), nn.Conv2d(48, 3, 1)).to(device)
    f1 = torch.tensor([[0, .5], [.5, 0]], device=device).reshape(1, 1, 2, 2).repeat(3, 1, 1, 1)
    f2 = torch.tensor([[.5, 0], [0, .5]], device=device).reshape(1, 1, 2, 2).repeat(3, 1, 1, 1)
    down = lambda t: (F.conv2d(t, f1, stride=2, groups=3), F.conv2d(t, f2, stride=2, groups=3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
    for _ in range(iters):
        n1, n2 = down(x)
        p1, p2 = n1 - net(n1), n2 - net(n2)
        loss_res = 0.5 * (F.mse_loss(n1, p2) + F.mse_loss(n2, p1))
        d1, d2 = down(x - net(x))
        loss = loss_res + 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        y = x - net(x)
        if pd > 1:
            y = F.pixel_shuffle(y.reshape(pd * pd, 3, 1, H // pd, W // pd).permute(2, 1, 0, 3, 4).reshape(1, 3 * pd * pd, H // pd, W // pd), pd)
        return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def score(task):
    index, a, b = task
    return {"index": index, "before": M.score(np.load(a), REFS[index], CORE), "after": M.score(np.load(b), REFS[index], CORE)}


def main():
    global REFS, CORE
    p = argparse.ArgumentParser()
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--rows", nargs="+", default=["ours_ttt"])
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pd", type=int, default=1, help="pixel-shuffle stride before ZS-N2N (1 = plain ZS-N2N)")
    p.add_argument("--every", type=int, default=1, help="keep image indices i with i %% every == 0 (6 -> 5 frames/video)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=["denoise", "score"], required=True)
    a = p.parse_args()
    ctx = M.Context("SDSD_indoor", a.low_receipt, a.opaque_manifest, a.gate_receipt)
    sources = M.row_sources(ctx, a.tuned_row)
    idx = [i for i in range(len(ctx.files)) if i % a.every == 0][: a.limit or None]
    if a.phase == "denoise":
        import torch
        torch.backends.cudnn.benchmark = True
        a.out.mkdir(parents=True, exist_ok=True)
        timing = {}
        for row_id in a.rows:
            manifest, _, root, _ = sources[row_id]
            d = a.out / row_id
            (d / "in").mkdir(parents=True, exist_ok=True)
            (d / "out").mkdir(parents=True, exist_ok=True)
            timing[row_id] = []
            for k, i in enumerate(idx):
                item = ctx.files[i]
                stem = Path(item["name"]).stem
                x = np.clip(M.load_output(root / stem / "output.pt.gz", manifest["rows"][i], item), 0, 1)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                y = denoise(x, a.iters, 0, "cuda:0", a.pd)
                torch.cuda.synchronize()
                timing[row_id].append({"name": item["name"], "seconds": time.perf_counter() - t,
                                       "peak_bytes": torch.cuda.max_memory_allocated()})
                np.save(d / "in" / f"{stem}.npy", x)
                np.save(d / "out" / f"{stem}.npy", y)
                print(row_id, k + 1, len(idx), round(timing[row_id][-1]["seconds"], 2), flush=True)
        with open(a.out / f"timing_{'_'.join(a.rows)}.json", "w") as f:
            json.dump({"label": LABEL, "iters": a.iters, "pd": a.pd, "indices": idx, "zsn2n_proto.py_sha256": M.sha256_file(__file__),
                       "gpu": torch.cuda.get_device_name(), "timing": timing}, f)
        return
    CORE = ctx.core
    REFS = {i: ctx.reference(i) for i in idx}
    result = {"label": LABEL, "zsn2n_proto.py_sha256": M.sha256_file(__file__), "iters": a.iters, "pd": a.pd, "indices": idx, "rows": {}}
    with get_context("fork").Pool(48) as pool:
        for row_id in a.rows:
            d = a.out / row_id
            tasks = [(i, d / "in" / f"{Path(it['name']).stem}.npy", d / "out" / f"{Path(it['name']).stem}.npy")
                     for i, it in ((i, ctx.files[i]) for i in idx)]
            result["rows"][row_id] = pool.map(score, tasks)
    with open(a.out / f"scores_{'_'.join(a.rows)}.json", "w") as f:
        json.dump(result, f, allow_nan=False)
    with open(a.out / f"scores_{'_'.join(a.rows)}.reads.json", "w") as f:
        json.dump(ctx.reads, f)
    for row_id, rs in result["rows"].items():
        print(row_id, {k: (float(np.mean([r[k]["psnr"] for r in rs])), float(np.mean([r[k]["rgb_ssim"] for r in rs])))
                       for k in ("before", "after")}, flush=True)


if __name__ == "__main__":
    main()

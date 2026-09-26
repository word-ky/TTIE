"""EXPLORATORY_ORACLE_NOT_FOR_PAPER -- T075-A task 2(a,b): per-image GT-oracle stop step.

Reruns frozen T070-A Ours-TTT trajectories (all 28 states) on SDSD-indoor through the T074-C derivation
machinery (``ours_tuning.derive``; frozen code objects, knob literals substituted), for the default knobs and
for the T074-C target-tuned knobs. Each run's selector output must reproduce the matching frozen row tensor
bit for bit. Every state is scored with the frozen T071-A metric (``metrics.score``); GT only via
``metrics.Context.reference``. Run from the frozen Ours source root with the frozen process environment.
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
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(K))
import metrics as M  # noqa: E402
import ours_tuning as OT  # noqa: E402

REFS = None
CORE = None


def score_step(args):
    index, img = args
    m = M.score(img, REFS[index], CORE)
    return m["psnr"], m["rgb_ssim"]


def main():
    global REFS, CORE
    OT.require(__debug__, "run without -O")
    p = argparse.ArgumentParser()
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--low-dir", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=28)
    a = p.parse_args()
    ctx = M.Context("SDSD_indoor", a.low_receipt, a.opaque_manifest, a.gate_receipt)
    CORE = ctx.core
    REFS = [ctx.reference(i) for i in range(len(ctx.files))]
    pool = get_context("fork").Pool(a.workers)  # before any CUDA init

    import torch
    from research_log.T070A.infer import FinalOurs
    from ttie.lolv2_gamma_core import native_rgb
    from ttie.common_gain import CommonRegion2
    from ttie.semantic_ttt import FixedObjective

    model = FinalOurs(a.manifest)
    binding = model.manifest["source_binding"]
    frozen, _ = M.load_json(ctx.rows_dir / "ours_ttt" / "output_manifest.json")
    tuned, _ = M.load_json(a.tuned_row / "output_manifest.json")
    settings = {"default": (OT.canonical_knobs({}), frozen["rows"]),
                "tuned": (OT.canonical_knobs(tuned["knobs"]), tuned["rows"])}
    lows = []
    for item in ctx.files:
        path = a.low_dir / item["name"]
        OT.require(M.sha256_file(path) == item["sha256"], f"low changed: {item['name']}")
        lows.append(native_rgb(path))
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"label": LABEL, "task": "T075A-oracle-steps", "gate_receipt_sha256": ctx.receipt_sha256,
              "oracle_steps.py_sha256": M.sha256_file(__file__), "ours_tuning.py_sha256": M.sha256_file(OT.__file__),
              "metrics.py_sha256": M.sha256_file(M.__file__), "execution_manifest_sha256": model.manifest_sha256,
              "settings": {}}
    for name, (knobs, rows) in settings.items():
        derived = OT.derive(knobs, binding)[0]
        imgdir = a.out / "images" / name
        imgdir.mkdir(parents=True)
        per_image = []
        for index, (item, low, row) in enumerate(zip(ctx.files, lows, rows)):
            x = low.to("cuda:0")
            objective = FixedObjective(model.scorer, x, OT.gate_receipt_for(model.gate, knobs))
            if not objective.active.any():
                with torch.no_grad():
                    states = CommonRegion2(objective.active).to(x)(x).detach().cpu()[None]
                selected, status = 0, "TTT_ABSTAIN_NO_ACTIVE_GATE"
            else:
                trace = derived["research_log.T062CR2.core.trajectory"](x, objective)
                decision, _, _ = derived["research_log.T070A.infer.select_trajectory"](x, trace, model.model)
                states, selected, status = trace["images"], decision["selected_step"], "TTT_EXECUTED"
            digest = M.sha256_bytes(states[selected].clone().contiguous().numpy().tobytes())
            OT.require(digest == row["output_tensor_sha256"], f"{name}: selector output does not reproduce row at {item['name']}")
            imgs = [np.ascontiguousarray(s[0].permute(1, 2, 0).numpy()) for s in states]
            scores = pool.map(score_step, [(index, im) for im in imgs])
            ps, ss = [s[0] for s in scores], [s[1] for s in scores]
            kp, ks = int(np.argmax(ps)), int(np.argmax(ss))
            stem = Path(item["name"]).stem
            np.save(imgdir / f"{stem}__psnr_oracle.npy", imgs[kp])
            np.save(imgdir / f"{stem}__ssim_oracle.npy", imgs[ks])
            per_image.append({"name": item["name"], "cluster": item["cluster"], "status": status,
                              "selected_step": int(selected), "psnr_steps": ps, "ssim_steps": ss,
                              "psnr_oracle_step": kp, "ssim_oracle_step": ks})
            print(f"{name} {index + 1}/{len(lows)} sel={selected} kp={kp} ks={ks}", flush=True)
        result["settings"][name] = {"knobs": knobs, "per_image": per_image}
    pool.close()
    with open(a.out / "oracle_steps.json", "x") as f:
        json.dump(result, f, allow_nan=False)
    with open(a.out / "reference_reads.json", "x") as f:
        json.dump(ctx.reads, f)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

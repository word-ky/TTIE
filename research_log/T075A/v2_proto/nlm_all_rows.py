"""T075-A task 3 fairness check (DEVELOPMENT-ON-TEST, SDSD-indoor): the same NLM post-filter on every row.

For every gated frozen row and the target-tuned Ours row: OpenCV fastNlMeansDenoisingColored (uint8 BGR,
h = hColor, template 7, search 21) at a few fixed strengths; h = 0 means the unfiltered frozen output.
Frozen T071-A metric via metrics.score; GT only via metrics.Context.reference. CPU only.
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

K = Path(os.environ.get("T074C_CODE", "/root/autodl-tmp/TTIE/T074C/code/research_log/T074C"))
sys.path.insert(0, str(K))
import metrics as M  # noqa: E402

LABEL = "DEVELOPMENT_ON_TEST_SDSD_INDOOR"
H = (4, 8, 12, 15, 18, 22)
REFS = CORE = None


def work(task):
    import cv2
    cv2.setNumThreads(1)
    index, row_id, path, row, item = task
    y = REFS[index]
    x = np.clip(np.asarray(M.load_output(path, row, item), np.float64), 0, 1)
    out = {"index": index, "row": row_id, "cluster": item["cluster"], "0": M.score(x, y, CORE)}
    bgr = np.ascontiguousarray(np.round(x[..., ::-1] * 255).astype(np.uint8))
    for h in H:
        d = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        out[str(h)] = M.score(d[..., ::-1].astype(np.float64) / 255.0, y, CORE)
    return out


def main():
    global REFS, CORE
    p = argparse.ArgumentParser()
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--opaque-manifest", type=Path, required=True)
    p.add_argument("--gate-receipt", type=Path, required=True)
    p.add_argument("--tuned-row", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    ctx = M.Context("SDSD_indoor", a.low_receipt, a.opaque_manifest, a.gate_receipt)
    CORE = ctx.core
    sources = M.row_sources(ctx, a.tuned_row)
    REFS = [ctx.reference(i) for i in range(len(ctx.files))]
    tasks = [(i, r, root / Path(item["name"]).stem / "output.pt.gz", man["rows"][i], item)
             for r, (man, _, root, _) in sources.items() if r != "ours_step0" for i, item in enumerate(ctx.files)]
    with get_context("fork").Pool(a.workers) as pool:
        recs = pool.map(work, tasks, chunksize=2)
    with open(a.out, "x") as f:
        json.dump({"label": LABEL, "nlm_all_rows.py_sha256": M.sha256_file(__file__), "h": H,
                   "gate_receipt_sha256": ctx.receipt_sha256, "records": recs}, f, allow_nan=False)
    with open(Path(str(a.out) + ".reads.json"), "x") as f:
        json.dump(ctx.reads, f)
    print("DONE", len(recs), flush=True)


if __name__ == "__main__":
    main()

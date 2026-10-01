import gzip, json, sys, numpy as np, torch, cv2
from pathlib import Path
L = Path("/root/autodl-tmp/TTIE/T074B/targets/LSRW"); R = Path("/root/autodl-tmp/TTIE/T075B/runs/LSRW")
files = json.load(open(L / "low_receipt.json"))["files"]
rows = ["promptir", "dctta", "retinexformer", "snr_aware", "ours_step0", "ours_ttt"]
out = []
for f in files:
    stem = Path(f["name"]).stem
    low = cv2.imread(str(L / "low" / f["name"]), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float64) / 255
    rec = {"name": f["name"], "hw": low.shape[:2], "low_mean": low.mean(), "low_max": low.max()}
    for r in rows:
        with gzip.open(R / r / stem / "output.pt.gz", "rb") as s:
            t = torch.load(s, map_location="cpu", weights_only=True)[0].double()
        rec[r] = float(t.clamp(0, 1).mean()); rec[r + "_raw_minmax"] = (float(t.min()), float(t.max()))
    out.append(rec)
for cam in ("Huawei", "Nikon"):
    sub = [o for o in out if o["name"].startswith(cam)]
    print(cam, len(sub), sub[0]["hw"], "low_mean %.4f" % np.mean([o["low_mean"] for o in sub]),
          " ".join("%s %.4f" % (r, np.mean([o[r] for o in sub])) for r in rows))
for o in out[:3] + out[30:33]:
    print(o["name"], "%.4f" % o["low_mean"], "%.3f" % o["low_max"], {r: round(o[r], 4) for r in rows}, o["promptir_raw_minmax"])

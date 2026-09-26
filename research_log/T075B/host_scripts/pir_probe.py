"""Scratch execution check (T075-B LSRW PromptIR): re-run the frozen static forward on selected lows with input dumps.
Writes nothing into frozen run dirs. Compares output tensor hash with the frozen manifest row."""
import sys, os, json, hashlib, torch, numpy as np, cv2
sys.path.insert(0, os.environ["T073B_SHARED"]); sys.path.insert(0, "/root/autodl-tmp/TTIE/T073B/source")
from low_only_loader import LowOnlyPromptTestDataset
from run_low_only import load_promptir
torch.manual_seed(23); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
low_dir, frozen, names = sys.argv[1], sys.argv[2], sys.argv[3:]
man = {r["low_name"]: r for r in json.load(open(os.path.join(frozen, "output_manifest.json")))["rows"]}
ds = LowOnlyPromptTestDataset(low_dir)
ck = "/root/autodl-tmp/TTIE/T073B/shared/epoch=80.ckpt"
print("checkpoint_sha256", hashlib.sha256(open(ck, "rb").read()).hexdigest())
model = load_promptir(ck).cuda().eval()
for n in names:
    i = ds.names.index(n); name, low = ds[i]
    ref = cv2.imread(os.path.join(low_dir, n), cv2.IMREAD_UNCHANGED)
    ref_rgb = torch.from_numpy(ref[..., ::-1].copy()).permute(2, 0, 1).float() / 255
    print(n, "input", tuple(low.shape), low.dtype, "min %.4f max %.4f mean %.4f" % (low.min(), low.max(), low.mean()),
          "per-channel mean", [round(float(c), 4) for c in low.mean((1, 2))],
          "equal_to_cv2_png_RGB/255:", bool(torch.equal(low, ref_rgb)), "png dtype/shape", ref.dtype, ref.shape)
    with torch.no_grad():
        out = model(low.unsqueeze(0).cuda()).cpu().contiguous()
    h = hashlib.sha256(out.numpy().tobytes()).hexdigest()
    print(n, "output mean %.4f min %.4f max %.4f" % (out.mean(), out.min(), out.max()),
          "residual(out-in) mean %.4f absmean %.4f" % ((out[0] - low).mean(), (out[0] - low).abs().mean()),
          "hash_matches_frozen:", h == man[n]["output_tensor_sha256"])

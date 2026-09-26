"""Scratch diagnostic: does PromptIR's near-identity behaviour on LSRW depend on geometry or on content?"""
import sys, os, torch, numpy as np, cv2
sys.path.insert(0, os.environ["T073B_SHARED"]); sys.path.insert(0, "/root/autodl-tmp/TTIE/T073B/source")
from run_low_only import load_promptir
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
model = load_promptir("/root/autodl-tmp/TTIE/T073B/shared/epoch=80.ckpt").cuda().eval()
def run(tag, rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        y = model(x.cuda()).cpu()
    print(f"{tag:48s} {tuple(x.shape[2:])} in_mean {x.mean():.4f} out_mean {y.clamp(0,1).mean():.4f} gain {y.clamp(0,1).mean()/x.mean():.2f}")
def rd(p): return cv2.imread(p)[..., ::-1]
LS = "/root/autodl-tmp/TTIE/T074B/targets/LSRW/low/"; SD = "/root/autodl-tmp/TTIE/T074B/targets/SDSD_indoor/low/"
for n in ("Huawei__2037.png", "Huawei__2050.png", "Nikon__3003.png"):
    a = rd(LS + n); run("LSRW " + n + " native", a)
    h = a.shape[0]; run("LSRW " + n + " center-crop 512x960", a[(h - 512) // 2:(h - 512) // 2 + 512])
    run("LSRW " + n + " resize 512x960", cv2.resize(a, (960, 512), interpolation=cv2.INTER_AREA))
for n in ("pair11__0177.png", "pair4__0188.png"):
    a = rd(SD + n); run("SDSD " + n + " native", a)
    run("SDSD " + n + " resize 720x960", cv2.resize(a, (960, 720), interpolation=cv2.INTER_LINEAR))
    run("SDSD " + n + " reflect-pad to 720x960", np.pad(a, ((104, 104), (0, 0), (0, 0)), mode="reflect"))

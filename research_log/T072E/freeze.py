"""Inference/freeze stage. It has no reference or metric input surface."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from PIL import Image

METHODS=("ours","retinexformer","snr_aware")
FORBIDDEN=("gt","reference","clean","normal","label","metric","psnr","ssim")
def _guard(path):
    parts={p.lower() for p in Path(path).parts}
    bad=parts.intersection(FORBIDDEN)
    if bad: raise PermissionError(f"reference path denied: {sorted(bad)}")
def canonical_lows(metadata):
    rows=json.loads(Path(metadata).read_text(encoding="utf-8"))["rows"]
    names=[r["name"] for r in rows]
    if len(names)!=150 or len(set(names))!=150 or names!=sorted(names): raise ValueError("canonical low declaration mismatch")
    return names
def low_probe(path):
    _guard(path)
    with Image.open(path) as im: return {"geometry":[im.width,im.height],"mode":im.mode,"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def build_freeze(rows, outputs, bindings):
    names=[r["name"] for r in rows]
    if len(names)!=150 or len(set(names))!=150: raise ValueError("duplicate/incomplete low set")
    if set(outputs)!=set(METHODS): raise ValueError("method coverage mismatch")
    images={}
    for method in METHODS:
        if set(outputs[method])!=set(names): raise ValueError(f"incomplete {method}")
        images[method]={n:outputs[method][n] for n in names}
    manifest={"schema":"ttie-uhdll-freeze-v1","pairs_manifest_sha256":"3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb","methods":list(METHODS),"bindings":bindings,"images":images,"real_reference_reads":0,"real_metrics":0}
    manifest["sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return manifest
def write_freeze(path, manifest):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8"); p.chmod(0o444)

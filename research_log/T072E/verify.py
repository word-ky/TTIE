from __future__ import annotations
import hashlib,json
from research_log.T072E.freeze import METHODS
def verify(manifest, expected_names):
    if manifest.get("schema")!="ttie-uhdll-freeze-v1" or manifest.get("pairs_manifest_sha256")!="3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb": return False
    if manifest.get("methods")!=list(METHODS) or manifest.get("real_reference_reads")!=0 or manifest.get("real_metrics")!=0: return False
    imgs=manifest.get("images",{})
    if set(imgs)!=set(METHODS): return False
    for m in METHODS:
        if set(imgs[m])!=set(expected_names): return False
        for x in imgs[m].values():
            if x.get("geometry") not in ([3840,2160],["3840","2160"]): return False
            if x.get("finite") is not True or not x.get("sha256"): return False
    core={k:v for k,v in manifest.items() if k!="sha256"}
    return hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest()==manifest.get("sha256")

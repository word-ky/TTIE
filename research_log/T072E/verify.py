from __future__ import annotations
import hashlib,json
from pathlib import Path
from research_log.T072E.freeze import METHODS
from research_log.T072E.constants import EXPECTED_BINDINGS
def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def verify(manifest, expected_names, artifact_root=None, authorized_artifacts=None):
    if manifest.get("schema")!="ttie-uhdll-freeze-v2" or manifest.get("pairs_manifest_sha256")!="3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb": return False
    if manifest.get("bindings")!=EXPECTED_BINDINGS or artifact_root is None or authorized_artifacts is None: return False
    if manifest.get("methods")!=list(METHODS) or manifest.get("real_reference_reads")!=0 or manifest.get("real_metrics")!=0: return False
    imgs=manifest.get("images",{})
    if set(imgs)!=set(METHODS): return False
    for m in METHODS:
        if set(imgs[m])!=set(expected_names): return False
        for x in imgs[m].values():
            if x.get("geometry") not in ([3840,2160],["3840","2160"]): return False
            if x.get("finite") is not True or not x.get("sha256") or not x.get("artifact_path"): return False
            rel=Path(x["artifact_path"])
            if rel.is_absolute() or ".." in rel.parts or rel.as_posix() not in authorized_artifacts: return False
            artifact=Path(artifact_root)/rel
            if not artifact.is_file() or _sha256(artifact)!=x["sha256"] or _sha256(artifact)!=authorized_artifacts[rel.as_posix()]: return False
    core={k:v for k,v in manifest.items() if k!="sha256"}
    return hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":")).encode()).hexdigest()==manifest.get("sha256")

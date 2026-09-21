"""Evaluation stage: separate process and requires a verified complete freeze."""
from __future__ import annotations
import json
from pathlib import Path
from research_log.T072E.freeze import METHODS
def require_freeze(path, expected_names, verify):
    p=Path(path); data=json.loads(p.read_text(encoding="utf-8"))
    if not verify(data,expected_names): raise ValueError("freeze manifest is not verifier-approved")
    if data.get("real_reference_reads",0)!=0: raise ValueError("freeze contains pre-freeze reference reads")
    return data
def evaluate(freeze_path, expected_names, verify, references):
    data=require_freeze(freeze_path,expected_names,verify)
    refs={n:references[n] for n in expected_names}
    return {m:{n: {"output":data["images"][m][n],"reference":refs[n]} for n in expected_names} for m in METHODS}

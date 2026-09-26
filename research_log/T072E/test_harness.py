import hashlib,json,pytest
from research_log.T072E.freeze import build_freeze,low_probe
from research_log.T072E.verify import verify
from research_log.T072E.evaluate import require_freeze
from research_log.T072E.constants import EXPECTED_BINDINGS
def rows(): return [{"name":f"{i:03d}.JPG"} for i in range(150)]
def outputs(rs,root):
    result={}
    for m in ("ours","retinexformer","snr_aware"):
        result[m]={}
        for r in rs:
            rel=f"{m}/{r['name']}.json"; p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((m+"-"+r["name"]).encode())
            result[m][r["name"]]={"artifact_path":rel,"geometry":[3840,2160],"finite":True,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    return result
def catalog(x): return {v["artifact_path"]:v["sha256"] for method in x["images"].values() for v in method.values()}
def make(tmp_path,bindings=None):
    rs=rows(); x=build_freeze(rs,outputs(rs,tmp_path),bindings or EXPECTED_BINDINGS); return rs,x,tmp_path,catalog(x)
def test_complete_and_verify(tmp_path):
    rs,x,root,cat=make(tmp_path); assert verify(x,[r["name"] for r in rs],root,cat)
@pytest.mark.parametrize("case",["missing","duplicate","altered_hash","altered_binding","wrong_geometry","incomplete","tamper"])
def test_rejects(case,tmp_path):
    rs,x,root,cat=make(tmp_path)
    if case in ("missing","incomplete"): del x["images"]["ours"][rs[0]["name"]]
    elif case=="duplicate": x["images"]["ours"]["extra.JPG"]=x["images"]["ours"][rs[0]["name"]]
    elif case=="altered_hash": x["images"]["ours"][rs[0]["name"]]["sha256"]="bad"
    elif case=="altered_binding":
        x["bindings"]["ours"]["scientific_source"]="changed"
        x["sha256"]=hashlib.sha256(json.dumps({k:v for k,v in x.items() if k!="sha256"},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    elif case=="wrong_geometry": x["images"]["ours"][rs[0]["name"]]["geometry"]=[1,1]
    elif case=="tamper": x["sha256"]="bad"
    assert not verify(x,[r["name"] for r in rs],root,cat)
def test_self_consistent_output_substitution_rejected(tmp_path):
    rs,x,root,cat=make(tmp_path); p=root/"ours"/f"{rs[0]['name']}.json"; p.write_bytes(b"substituted"); x["images"]["ours"][rs[0]["name"]]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest(); x["sha256"]=hashlib.sha256(json.dumps({k:v for k,v in x.items() if k!="sha256"},sort_keys=True,separators=(",",":")).encode()).hexdigest(); assert not verify(x,[r["name"] for r in rs],root,cat)
def test_reference_guard():
    with pytest.raises(PermissionError): low_probe("/tmp/testing_set/gt/1.JPG")
def test_evaluation_requires_complete(tmp_path):
    rs=rows(); x=build_freeze(rs,outputs(rs,tmp_path),{"ours":"x","retinexformer":"y","snr_aware":"z"}); del x["images"]["ours"][rs[0]["name"]]
    p=tmp_path/"freeze.json"; p.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(ValueError): require_freeze(p,[r["name"] for r in rs],verify,tmp_path,{})

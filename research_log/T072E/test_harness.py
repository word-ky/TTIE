import json,pytest
from research_log.T072E.freeze import build_freeze,low_probe
from research_log.T072E.verify import verify
from research_log.T072E.evaluate import require_freeze
def rows(): return [{"name":f"{i:03d}.JPG"} for i in range(150)]
def outputs(rs): return {m:{r["name"]:{"geometry":[3840,2160],"finite":True,"sha256":f"{m}-{r['name']}"} for r in rs} for m in ("ours","retinexformer","snr_aware")}
def test_complete_and_verify():
    rs=rows(); x=build_freeze(rs,outputs(rs),{"ours":"e7f129d","retinexformer":"T071B","snr_aware":"T071B"}); assert verify(x,[r["name"] for r in rs])
@pytest.mark.parametrize("case",["missing","duplicate","altered_hash","altered_binding","wrong_geometry","incomplete","tamper"])
def test_rejects(case):
    rs=rows(); x=build_freeze(rs,outputs(rs),{"ours":"e7f129d","retinexformer":"T071B","snr_aware":"T071B"})
    if case in ("missing","incomplete"): del x["images"]["ours"][rs[0]["name"]]
    elif case=="duplicate": x["images"]["ours"]["extra.JPG"]=x["images"]["ours"][rs[0]["name"]]
    elif case=="altered_hash": x["images"]["ours"][rs[0]["name"]]["sha256"]="bad"
    elif case=="altered_binding": x["bindings"]["ours"]="changed"
    elif case=="wrong_geometry": x["images"]["ours"][rs[0]["name"]]["geometry"]=[1,1]
    elif case=="tamper": x["sha256"]="bad"
    assert not verify(x,[r["name"] for r in rs])
def test_reference_guard():
    with pytest.raises(PermissionError): low_probe("/tmp/testing_set/gt/1.JPG")
def test_evaluation_requires_complete(tmp_path):
    rs=rows(); x=build_freeze(rs,outputs(rs),{"ours":"x","retinexformer":"y","snr_aware":"z"}); del x["images"]["ours"][rs[0]["name"]]
    p=tmp_path/"freeze.json"; p.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(ValueError): require_freeze(p,[r["name"] for r in rs],verify)

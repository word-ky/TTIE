import subprocess,sys
import pytest
from research_log.T059X.core import classify,validate_inputs,sha

def test_frozen_boundaries():
    assert 'insufficient' in classify(49,-1,-1,100,1)
    assert 'transfers aggregate' in classify(50,-1,-1,55,0)
    assert 'does not transfer' in classify(100,0,-1,55,0)
    assert 'does not transfer' in classify(100,-1,-1,54,0)
def test_mutated_input(tmp_path):
    p=tmp_path/'low';p.write_bytes(b'a');inputs={str(p):sha(p)};p.write_bytes(b'b')
    with pytest.raises(AssertionError):validate_inputs(inputs)
def test_firewall(tmp_path):
    low=tmp_path/'low';low.write_text('ok');normal=tmp_path/'normal';normal.write_text('forbidden');out=tmp_path/'out';out.mkdir()
    program='from research_log.T059X.core import install_firewall;from pathlib import Path;install_firewall('+repr(str(tmp_path))+',['+repr(str(low))+'],'+repr(str(out))+');assert Path('+repr(str(low))+').read_text()=="ok";Path('+repr(str(normal))+').read_text()'
    r=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert r.returncode!=0 and 'outside bound low-only inputs' in r.stderr

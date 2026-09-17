import builtins,torch
from research_log.T059C2.core import split_manifest,read_reference_rows,numeric_gates

def test_noncontiguous_image_groups():
    banks=[dict(image_id=i,states=18+(layer*80+i<146)) for layer in range(5) for i in range(80)]
    s=split_manifest(banks)
    assert s['image_ids']['heldout']==list(range(0,80,5))
    assert s['bank_indices']['heldout']==[i for i in range(400) if i%80%5==0]
    assert len(s['row_indices']['train'])+len(s['row_indices']['heldout'])==7346

def test_mixed_chunk_reads_only_authorized_storage_rows(tmp_path,monkeypatch):
    path=tmp_path/'chunk.pt';x=torch.arange(4*64,dtype=torch.float32).reshape(4,1,1,8,8);torch.save(dict(indices=[20,21,22,23],g_R=x),path)
    real=builtins.open;reads=[]
    class Traced:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*a):self.f.close()
        def seek(self,*a):return self.f.seek(*a)
        def read(self,n):
            pos=self.f.tell();b=self.f.read(n);reads.append((pos,len(b)));return b
    monkeypatch.setattr(builtins,'open',lambda *a,**k:Traced(real(*a,**k)))
    result,ranges=read_reference_rows(path,{21,23})
    assert list(result)==[21,23] and torch.equal(result[21],x[1]) and torch.equal(result[23],x[3])
    assert reads[1:]==[(r['offset'],256) for r in ranges] and len(reads)==3

def test_numeric_verdict_ignores_no_stored_flags():
    good=dict(positive_fraction=.99,median_cosine=.95,value_huber=.07)
    assert numeric_gates(good,dict(positive_fraction=.9,median_cosine=.6))['classification'].endswith('supported')
    assert numeric_gates(good,dict(positive_fraction=.9,median_cosine=.49))['classification'].endswith('not supported under fixed split')

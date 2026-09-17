import builtins,torch
from research_log.T059E.core import nested_split,selected_rows
from research_log.T059E.fit import relative_value

def test_nested_noncontiguous_partition():
    banks=[dict(image_id=i,states=18+(layer*80+i<146)) for layer in range(5) for i in range(80)]
    s=nested_split(banks)
    assert s['image_ids']['outer']==list(range(0,80,5)) and s['image_ids']['heldout']==list(range(1,80,5))
    assert len(s['image_ids']['train'])==48 and len(s['bank_indices']['train'])==240
    assert sorted(sum(s['row_indices'].values(),[]))==list(range(7346))

def test_sparse_mixed_cache_never_reads_excluded_numeric_ranges(tmp_path,monkeypatch):
    p=tmp_path/'mixed.pt';j=torch.arange(4*28*64,dtype=torch.float32).reshape(4,28,64);g=torch.arange(256,dtype=torch.float32).reshape(4,1,1,8,8)
    torch.save(dict(indices=[0,1,2,3],J=j,g_R=g),p);real=builtins.open;reads=[]
    class Traced:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*a):self.f.close()
        def seek(self,*a):return self.f.seek(*a)
        def read(self,n):
            pos=self.f.tell();b=self.f.read(n);reads.append((pos,len(b)));return b
    monkeypatch.setattr(builtins,'open',lambda *a,**k:Traced(real(*a,**k)))
    for key,shape,expected in [('J',(28,64),j),('g_R',(1,1,8,8),g)]:
        reads.clear();got,r=selected_rows(p,{2,3},key,shape)
        assert list(got)==[2,3] and torch.equal(got[2],expected[2]) and torch.equal(got[3],expected[3])
        assert reads[1:]==[(v['offset'],v['bytes']) for v in r['ranges']] and len(reads)==3

def test_relative_value_anchor_gradient_and_shared_offset_invariance():
    weight=torch.tensor(.3,requires_grad=True);bias=torch.tensor(4.,requires_grad=True);x=torch.tensor([2.,1.]);anchor=torch.tensor([1.,1.]);target=torch.tensor([.8,.2]);ta=torch.tensor([.2,.2])
    loss=relative_value(weight*x+bias,weight*anchor+bias,target,ta);loss.backward()
    assert abs(float(weight.grad)+.15)<1e-6 and float(bias.grad)==0
    assert abs(relative_value(weight.detach()*x+bias.detach()+10,weight.detach()*anchor+bias.detach()+10,target+8,ta+8).item()-loss.detach().item())<1e-6
    assert float(relative_value(torch.tensor([3.]),torch.tensor([3.]),torch.tensor([8.]),torch.tensor([8.])))==0


def test_fixed_relative_training_smoke_and_train_only_normalization():
    from research_log.T059E.fit import train_fixed_relative
    torch.manual_seed(23);x=torch.randn(8,28);mse=torch.rand(8,dtype=torch.float64)+.01
    records=[dict(jacobian=torch.randn(8,28,n),reference_gradient=torch.randn(8,n),direction_mask=torch.ones(8,dtype=torch.bool)) for n in [8,64]]
    head,history,initial=train_fixed_relative(x,mse,*records,torch.tensor([0,0,0,0,4,4,4,4]))
    assert len(history)==100 and all(torch.isfinite(v).all() for v in head.state_dict().values())
    assert torch.equal(head.x_mean,x.double().mean(0).float())
    assert torch.equal(head.y_mean,(mse.double()+1e-6).log().mean().float())

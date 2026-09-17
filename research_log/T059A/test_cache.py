import torch,pytest
from research_log.T059A.support import detail_jacobian,reconstruct,compare,save_chunk,reopen_cache,atomic_json,thash

def test_exact_vjps_chain_and_persistence(tmp_path):
    torch.manual_seed(7);v=torch.randn(1,1,8,8,requires_grad=True);w=torch.randn(8,64)
    phi=torch.cat([torch.ones(12),torch.sin(w@v.flatten()),torch.zeros(8)])
    head=torch.nn.Sequential(torch.nn.Linear(28,7),torch.nn.SiLU(),torch.nn.Linear(7,1)).requires_grad_(False)
    jac=detail_jacobian(phi,v);direct=torch.autograd.grad(head(phi).sum(),v)[0];q,recon=reconstruct(head,phi,jac)
    assert torch.allclose(jac[12:20],torch.cos(w@v.flatten())[:,None]*w)
    assert torch.count_nonzero(jac[:12])==torch.count_nonzero(jac[20:])==0
    check=compare(recon,direct);assert check['allclose'] and check['zero_status_matches']
    cache=dict(phi=phi.detach(),J=jac.detach(),q=q,reconstructed=recon,accepted=direct)
    row=dict(index=0,chain=check,**{k+'_sha256':thash(t) for k,t in cache.items()})
    chunk=save_chunk(tmp_path,[0],[cache]);atomic_json(tmp_path/'manifest.json',dict(rows=[row],chunks=[chunk]))
    assert reopen_cache(tmp_path,1)['all_tensor_hashes_verified']
    p=tmp_path/chunk['file'];p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(AssertionError):reopen_cache(tmp_path,1)

def test_fixed_gate_zero_and_failure():
    zero=torch.zeros(1,1,8,8);assert compare(zero,zero)['zero_status_matches'] and compare(zero,zero)['allclose']
    assert not compare(zero+3e-6,zero)['allclose']
    assert not compare(zero+1e-8,zero)['zero_status_matches']

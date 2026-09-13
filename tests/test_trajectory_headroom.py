import numpy as np
import torch
from ttie.semantic_ttt import Region2
from ttie.trajectory_headroom import render,saturation,summarize


def test_render_exact_saved_raw_and_inactive_region():
    image=torch.rand(1,3,31,33)
    active=torch.tensor([True,False,True,True]);original=Region2(active)
    with torch.no_grad():original.raw.copy_(torch.linspace(-.2,.2,8).reshape(1,2,2,2))
    frozen=original(image).detach().clone();state=original.raw.detach().clone()
    recovered=render(Region2(active),image,state)
    assert torch.equal(frozen,recovered)
    assert torch.equal(recovered[:,:,:15,16:],image[:,:,:15,16:])


def test_bounds_union_counts_collapsed_coordinates_once():
    lower=torch.zeros(1,2,2,2);upper=torch.ones_like(lower);upper[0,0,0,0]=0
    grid=lower.clone();grid[0,1]=.5
    s=saturation(grid,lower,upper)
    assert s['ev_either']==1 and s['ev_upper']==.25 and s['gamma_either']==0


def test_separate_oracles_fixed_steps_gaps_and_histograms():
    rows=[]
    for i in range(100):
        for k in range(41):
            rows.append(dict(index=i,low=str(i),step=k,psnr=100-(k-20)**2,ssim=.5-abs(k-30)*.01,
                        **{f'{c}_{b}':0 for c in ['ev','gamma'] for b in ['lower','upper','either']}))
    s,f,p=summarize(rows,[10]*100)
    assert len(f)==41 and len(p)==100
    assert s['best_global_fixed_step']=={'psnr':20,'ssim':30}
    assert s['headroom']['psnr_oracle']['psnr']['mean']==100
    assert np.isclose(s['headroom']['ssim_oracle']['ssim']['mean'],.2)
    assert s['step_histograms']['psnr_oracle'][20]==100
    assert s['step_histograms']['ssim_oracle'][30]==100

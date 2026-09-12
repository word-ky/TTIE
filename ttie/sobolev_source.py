"""T014 source-only derivative records; never imported by test-time energy code."""
import torch
from .semantic_ttt import FixedObjective,Region2
from .energy_bank import state_bank
from .energy_model import features

JACOBIAN=dict(shape='rows,28,8',coordinates='raw EV TL TR BL BR; raw gamma TL TR BL BR',
    renderer='T011 Region2(original active mask), re-render fixed bank raw states',
    forward_mode='active rows use grad-enabled CLIP for both cached features and Jacobians',
    projection='not differentiated',constants='first12 feature rows zero',
    method='exact reverse-mode autograd; eight CLIP evidence VJPs plus physical-grid derivatives',
    no_active='identity, value only; zero J and reference gradient',
    legacy_pixels='retain per-row max absolute difference from inherited T013 bank')


def source_bank(image,scorer,receipt,*,semantic_steps=40):
    bank=state_bank(image,scorer,receipt,semantic_steps=semantic_steps)
    obj=FixedObjective(scorer,image,receipt);model=Region2(obj.active).to(image)
    differences=[]
    for i,raw in enumerate(bank['states']):
        with torch.no_grad():model.raw.copy_(raw.to(image))
        output=model(image) if obj.active.any() else image
        scores=scorer(output);grid=model.physical_grid()[:,:2]
        differences.append(float((output.detach().cpu()-bank['images'][i]).abs().max()))
        for key,value in dict(images=output,scores=scores,grids=grid,features=features(obj,scores,grid)).items():
            bank[key][i].copy_(value.detach().cpu())
    bank['inherited_pixel_max_abs_differences']=differences
    return bank


def derivative_record(image,clean,obj,raw):
    model=Region2(obj.active).to(image)
    with torch.no_grad():model.raw.copy_(raw.to(image))
    output=model(image);scores=obj.scorer(output);grid=model.physical_grid()[:,:2]
    f=features(obj,scores,grid)
    jac=image.new_zeros(28,8)
    # Separate grid VJPs avoid traversing CLIP with a zero adjoint.
    for j in range(12,20):
        jac[j]=torch.autograd.grad(f[j],model.raw,retain_graph=True)[0].flatten()
    for j in range(8):
        jac[20+j]=torch.autograd.grad(grid.flatten()[j],model.raw,retain_graph=True)[0].flatten()
    y=((output-clean.to(image)).square().mean()+1e-6).log()
    truth=torch.autograd.grad(y,model.raw)[0]
    return dict(features=f.detach().cpu(),jacobian=jac.detach().cpu(),reference_gradient=truth.detach().cpu().flatten(),
                reference_log_mse=y.detach().cpu())


def source_derivatives(image,clean,scorer,receipt,bank):
    obj=FixedObjective(scorer,image,receipt);active=bool(obj.active.any());n=len(bank['states'])
    records=[]
    if active:
        for i,raw in enumerate(bank['states']):
            record=derivative_record(image,clean,obj,raw)
            assert torch.allclose(record['features'],bank['features'][i],atol=1e-6,rtol=1e-6)
            records.append(record)
    else:
        records=[dict(features=bank['features'][0],jacobian=torch.zeros(28,8),reference_gradient=torch.zeros(8),
                      reference_log_mse=((bank['images'][0]-clean.cpu()).square().mean()+1e-6).log())]
    result={k:torch.stack([r[k] for r in records]) for k in records[0]}
    result['active']=torch.full((n,),active,dtype=torch.bool)
    result['direction_mask']=result['active'] & (result['reference_gradient'].norm(dim=1)>0)
    return result

"""T030-A fixed low-only learned-gradient guard. No image/reference/metric IO."""
import torch

SPEC=dict(anchor_step=10,cosine_threshold=0.,search_steps=[11,40],norm_epsilon=1e-12,
          selection='minimum predicted energy over0..cutoff; earliest tie',
          fallback='original minimum-energy selector if anchor or visited comparison norm<=1e-12')

def select(energies,gradients,active):
    baseline=min(range(len(energies)),key=lambda s:(energies[s],s))
    mask=active.flatten().bool().repeat(2)
    vectors=gradients.detach().double().reshape(len(gradients),-1)[:,mask]
    norms=vectors.norm(dim=1);cutoff=len(energies)-1;crossing=None;comparisons=[];fallback=None
    if len(energies)<=10 or norms[10]<=SPEC['norm_epsilon']:
        fallback='degenerate_anchor'
    else:
        anchor=vectors[10]
        for step in range(11,len(energies)):
            if norms[step]<=SPEC['norm_epsilon']:
                fallback='degenerate_comparison';cutoff=len(energies)-1;break
            cosine=float(torch.dot(vectors[step],anchor)/(norms[step]*norms[10]))
            comparisons.append(dict(step=step,cosine=cosine))
            if cosine<=0.:
                crossing=step;cutoff=step-1;break
    chosen=baseline if fallback else min(range(cutoff+1),key=lambda s:(energies[s],s))
    return dict(original_step=baseline,selected_step=chosen,cutoff=cutoff,crossing=crossing,
                fallback=fallback,active_coordinates=int(mask.sum()),norms=norms.cpu().tolist(),comparisons=comparisons)

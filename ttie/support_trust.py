"""T032-A source-feature geometry and fixed first-exit selection; no image IO."""
import torch


def cross_image_distances(features, image_ids, x_mean, x_scale):
    z=(features.double()-x_mean.double())/x_scale.double()
    ids=image_ids.to(z.device)
    result=[]
    for start in range(0,len(z),128):
        q=z[start:start+128]
        d=torch.cdist(q,z,compute_mode='donot_use_mm_for_euclid_dist')
        d.masked_fill_(ids[start:start+len(q),None]==ids[None,:],float('inf'))
        result.append(d.min(1).values/(z.shape[1]**.5))
    return torch.cat(result)


def select(energies, distances, radius):
    original=min(range(len(energies)),key=lambda s:(energies[s],s))
    exit_step=next((i for i,d in enumerate(distances) if d>radius),None)
    cutoff=len(energies)-1 if exit_step is None else max(0,exit_step-1)
    chosen=min(range(cutoff+1),key=lambda s:(energies[s],s))
    return dict(original_step=original,selected_step=chosen,cutoff=cutoff,exit_step=exit_step)


def bind_radius(radius_receipt, spec, actual_freeze_hash):
    assert spec['radius_freeze_sha256']==actual_freeze_hash, 'radius freeze mismatch'
    assert spec['r_support']==radius_receipt['r_support'], 'radius value mismatch'
    assert spec['percentile']==.95 and spec['quantile_method']=='linear'
    return radius_receipt['r_support']

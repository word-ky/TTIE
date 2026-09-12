"""T012 passive observation and selection of the unchanged T011 trajectory."""
import copy
import torch
from .semantic_ttt import FixedObjective
from .projected_ttt import run_candidate

QUADRANTS=('TL','TR','BL','BR')
FEATURE_NAMES=[f'{group}_{q}' for group in ('active','signed_winner','original_winning_evidence','z_dark','z_bright','ev','gamma') for q in QUADRANTS]
FEATURE_NAMES+=['step_over_40','semantic_loss','loss_minus_previous','outgoing_gradient_norm','prior_projection_hit_fraction']
SCHEMA=dict(version='T012-33-v1',names=FEATURE_NAMES,dimension=33)


def feature_vectors(gate,scores,grids,diagnostics,calibration):
    active=torch.tensor(gate['active'],dtype=torch.float64)
    winner=torch.tensor(gate['winner'])
    signed=active*torch.where(winner==0,1.,-1.)
    evidence=torch.tensor(gate['evidence'],dtype=torch.float64)
    z=(scores.double()-torch.tensor(calibration['tau'],dtype=torch.float64))/torch.tensor(calibration['scale'],dtype=torch.float64)
    losses=diagnostics['loss_trajectory'];gradients=diagnostics['gradient_norms'];hit_count=0;features=[]
    for i in range(len(scores)):
        if i:
            projection=diagnostics['projections'][i-1]
            hit_count+=int((torch.tensor(projection['pre_raw'])!=torch.tensor(projection['post_raw'])).any())
        scalars=torch.tensor([i/40,losses[i],losses[i]-losses[i-1] if i else 0.,
                              gradients[i] if i<len(gradients) else 0.,hit_count/i if i else 0.],dtype=torch.float64)
        features.append(torch.cat((active,signed,evidence,z[i,:,0],z[i,:,1],grids[i,0,0].flatten(),grids[i,0,1].flatten(),scalars)).float())
    return torch.stack(features)


def capture(image,scorer,receipt,*,max_steps=40):
    frames=[];scores=[]
    def observed(pixels):
        value=scorer(pixels)
        frames.append(pixels.detach().cpu());scores.append(value.detach().cpu())
        return value
    objective=FixedObjective(observed,image,receipt)
    final=run_candidate(image,objective,'region2_ttt_projected',max_steps=max_steps,record_states=True)
    gate=dict(scores=objective.original_scores.cpu().tolist(),active=objective.active.cpu().tolist(),
              winner=objective.winner.cpu().tolist(),evidence=objective.evidence.cpu().tolist())
    if objective.active.any():frames=frames[1:];scores=scores[1:]
    # The recorder adds no model evaluation, optimizer update or reference input.
    images=torch.stack(frames);current_scores=torch.stack(scores)
    grid0=torch.zeros_like(final['grid'].cpu());grid0[:,1]=1.
    grids=torch.stack([grid0]+[torch.tensor(p['post_grid']) for p in final['diagnostics']['projections']])
    states=torch.stack(final['states'])
    features=feature_vectors(gate,current_scores,grids,final['diagnostics'],receipt['calibration'])
    return dict(images=images,scores=current_scores,grids=grids,states=states,features=features,gate=gate,
                diagnostics=final['diagnostics'])


def checkpoint(trajectory,index):
    index=min(index,len(trajectory['images'])-1)
    d=copy.deepcopy(trajectory['diagnostics']);d['trajectory_updates']=d['steps']
    d['selected_step']=index;d['steps']=index;d['stop_reason']='saved_checkpoint'
    d['loss_after']=d['loss_trajectory'][index];d['loss_trajectory']=d['loss_trajectory'][:index+1]
    d['gradient_norms']=d['gradient_norms'][:index];d['gradient_vectors']=d['gradient_vectors'][:index]
    d['projections']=d['projections'][:index];d['parameter_ranges']=d['parameter_ranges'][:index+1]
    grid=trajectory['grids'][index]
    d['final_grid']=grid.tolist();d['final_ranges']=dict(min=grid.amin(dim=(0,2,3)).tolist(),max=grid.amax(dim=(0,2,3)).tolist())
    return dict(image=trajectory['images'][index],raw=trajectory['states'][index],grid=grid,
                states=list(trajectory['states'][:index+1]),diagnostics=d)


def select_checkpoint(trajectory,head):
    if not any(trajectory['gate']['active']):return dict(selected_step=0,scores=[None],bypass='no_active')
    with torch.no_grad():scores=head(trajectory['features']).flatten().cpu()
    return dict(selected_step=int(scores.argmin()),scores=scores.tolist(),bypass=None)


def select_image(image,scorer,receipt,head,*,max_steps=40):
    trajectory=capture(image,scorer,receipt,max_steps=max_steps)
    decision=select_checkpoint(trajectory,head)
    return checkpoint(trajectory,decision['selected_step']),trajectory,decision

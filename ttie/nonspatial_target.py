"""T020-B development/reference-only target audit of frozen cached T014 states."""
import argparse
from pathlib import Path
import statistics
import torch
from .local_geometry import now, sha, write, ratio
from .natural import load_image
from .soft_basis.renderer import FixedCorners
from .routing.provenance import verify_source
import json

LOCK=Path('research_log/T020B_pipeline_lock.json')
INPUTS=Path('research_log/T020B_source_inputs')
CONDITIONS=('clean','homogeneous_dark','homogeneous_bright')
HARD=tuple((x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6))
CROSS=(4,1,7,3,5)
DELTA=.01
METHOD='region2_ttt_energy_sobolev'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def axis(center, lower, upper):
    # Same positive-denominator implementation as accepted T019-A. With a
    # >=1% gain center cannot win a minimum tie; lower precedes upper.
    if center==0:
        return dict(value=.5,gains=[None,None],threshold_equal=False,
                    noncenter_tie=lower==upper,moving_tie=False)
    gains=((center-lower)/center,(center-upper)/center)
    moves=max(gains)>=DELTA
    value=(.4 if lower<=upper else .6) if moves else .5
    return dict(value=value,gains=list(gains),threshold_equal=max(gains)==DELTA,
                noncenter_tie=lower==upper,moving_tie=moves and lower==upper)


def target(values):
    cross=[values[i] for i in CROSS]
    x=axis(cross[0],cross[1],cross[2]);y=axis(cross[0],cross[3],cross[4])
    bx,by=x['value'],y['value'];index=HARD.index((bx,by,0.))
    labels={.5:'center',.4:'lower',.6:'upper'}
    return dict(cross_mse=cross,x=x,y=y,x_label=labels[bx],y_label=labels[by],bx=bx,by=by,
        hard_index=index,H0=values[4],H_delta=values[index],H_star=min(values),
        movement='no_move' if bx==.5 and by==.5 else 'x_only' if by==.5 else 'y_only' if bx==.5 else 'both',
        center_in_oracle_tie_set=values[4]==min(values))


def summarize(rows):
    groups={}
    for name in ('nonspatial_pool',*CONDITIONS):
        group=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','H_delta','H_star')}
        groups[name]=dict(count=len(group),mse=m,ratios=dict(H_delta_over_H0=ratio(m['H_delta'],m['H0']),H_star_over_H0=ratio(m['H_star'],m['H0'])),
            outcomes=dict(beneficial=sum(r['H_delta']<r['H0'] for r in group),equal=sum(r['H_delta']==r['H0'] for r in group),harmful=sum(r['H_delta']>r['H0'] for r in group)),
            movements={k:sum(r['movement']==k for r in group) for k in ('no_move','x_only','y_only','both')},
            labels={a:{k:sum(r[a+'_label']==k for r in group) for k in ('center','lower','upper')} for a in ('x','y')},
            center_in_oracle_tie_count=sum(r['center_in_oracle_tie_set'] for r in group),
            center_in_oracle_tie_fraction=statistics.mean(r['center_in_oracle_tie_set'] for r in group),
            harmful_rows=[r['row_index'] for r in group if r['H_delta']>r['H0']])
    clauses={name+'_safety':g['mse']['H_delta']<=1.01*g['mse']['H0'] for name,g in groups.items()}
    clauses['zero_harmful']=groups['nonspatial_pool']['outcomes']['harmful']==0
    return dict(groups=groups,clauses=clauses,acceptance_vector=list(clauses.values()),passed=sum(clauses.values()),
        target_viable=all(clauses.values()),reference_only=True,fresh_qualified=False,
        verdict='T020-B development target positive' if all(clauses.values()) else 'T020-B development target negative')


@torch.no_grad()
def run(output,source,device):
    lock=read(LOCK)
    paths=list(lock['baseline_code_sha256'])+['ttie/nonspatial_target.py',LOCK.as_posix(),'research_log/T020B_verify.py','scripts/run_t020b_a6000.sh']
    paths += [(INPUTS/name).as_posix() for name in lock['origins']]
    code=verify_source(source,paths)
    for path,h in lock['baseline_code_sha256'].items():assert code[path]==h
    for name,o in lock['origins'].items():assert sha(INPUTS/name)==o['sha256']
    cache=Path(lock['accepted_cache']);images=Path(lock['images'])
    for name,local in [('artifact_manifest.json','accepted_cache_manifest.json'),('config.json','accepted_cache_config.json')]:assert sha(cache/name)==lock['origins'][local]['sha256']
    manifest=read(INPUTS/'development_manifest.json');entries=read(INPUTS/'accepted_cache_manifest.json')
    by={(e['image_id'],e['condition']):e for e in entries}
    assert [e['image_id'] for e in manifest['images']]==lock['development_ids'] and len(lock['development_ids'])==40
    assert lock['conditions']==list(CONDITIONS) and lock['candidate_coordinates']==[list(c) for c in HARD] and lock['delta']==DELTA
    torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True);started=now()
    write(output/'config.json',dict(task='T020-B',source_sha=source,source_code_sha256=code,pipeline_lock_sha256=sha(LOCK),
        lock=lock,device=device,torch_version=torch.__version__,reference_only=True,new_image_ids=0,training=False,adaptation_rerun=False))
    table=[];cache_bindings=[]
    for image_entry in manifest['images']:
        path=images/image_entry['filename'];assert sha(path)==image_entry['sha256']
        clean=load_image(path)
        for condition in CONDITIONS:
            old=by[(image_entry['image_id'],condition)];folder=cache/old['directory'];bound={}
            for method,names in [('semantic',('outputs.pt',)),(METHOD,('outputs.pt','trajectory.pt','decisions.json'))]:
                for name in names:
                    f=folder/method/name;h=sha(f);assert h==old['files'][method][name]['sha256'];bound[method+'/'+name]=h
            image=torch.load(folder/'semantic/outputs.pt',weights_only=True,map_location='cpu')['identity']['image'].to(device)
            original=torch.load(folder/METHOD/'outputs.pt',weights_only=True,map_location='cpu')[METHOD]
            trace=torch.load(folder/METHOD/'trajectory.pt',weights_only=True,map_location='cpu');decision=read(folder/METHOD/'decisions.json')
            step=decision['selection']['selected_step'];assert torch.equal(original['grid'],trace['grids'][step])
            corners=original['grid'].to(device)
            canonical=FixedCorners(corners,(.5,.5,0.))(image).cpu()
            assert torch.equal(canonical,original['image']), 'Cached canonical pixels must reproduce exactly'
            values=[float((FixedCorners(corners,c)(image).cpu()-clean).square().mean()) for c in HARD]
            assert values[4]==float((original['image']-clean).square().mean())
            i=len(table);table.append(dict(row_index=i,image_id=image_entry['image_id'],condition=condition,
                source_directory=old['directory'],source_image_sha256=image_entry['sha256'],canonical_selected_step=step,candidate_mse=values))
            state=output/f'{i:03d}';state.mkdir();torch.save(dict(corners=original['grid']),state/'state.pt')
            cache_bindings.append(dict(row_index=i,source_directory=old['directory'],files_sha256=bound,
                corners_file_sha256=sha(state/'state.pt'),canonical_pixels_exact=True))
            print('Cached canonical exact; rendered',i+1,'/120',flush=True)
    assert len(table)==120
    write(output/'candidate_table.json',table);write(output/'cache_bindings.json',cache_bindings)
    write(output/'table_frozen.json',dict(started_utc=started,finalized_utc=now(),source_sha=source,reference_only=True,
        candidate_table_sha256=sha(output/'candidate_table.json'),cache_bindings_sha256=sha(output/'cache_bindings.json'),config_sha256=sha(output/'config.json'),episodes=120,canonical_exact=120))
    rows=[dict(**r,**target(r['candidate_mse'])) for r in table]
    write(output/'evaluation.json',rows);summary=summarize(rows);write(output/'summary.json',summary)
    write(output/'report_receipt.json',dict(completed_utc=now(),reference_only=True,
        **{n+'_sha256':sha(output/(n+'.json')) for n in ('candidate_table','table_frozen','evaluation','summary','config','cache_bindings')}))
    print(summary['verdict'],summary['passed'],'/5',summary['clauses'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(a.output,a.source_sha,a.device)

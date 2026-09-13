"""T020-C: cached label-free features, unchanged OOF learner, post-freeze join."""
import argparse
import json
import platform
import statistics
from pathlib import Path
import torch
from .local_geometry import now, sha, write, ratio
from .direction_probe import RECIPE, axis_features
from .direction_probe_run import train_fold, freeze_oof
from .deadband_probe import training_targets
from .routing.provenance import verify_source

LOCK = Path('research_log/T020C_lock.json')
INPUTS = Path('research_log/T020C_inputs')
CROSS = (4, 1, 7, 3, 5)
HARD = tuple((x, y, 0.) for x in (.4, .5, .6) for y in (.4, .5, .6))
CONDITIONS = ('clean', 'homogeneous_dark', 'homogeneous_bright')
METHOD = 'region2_ttt_energy_sobolev'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def preflight(source):
    lock = read(LOCK)
    paths = read('research_log/T020C_source_paths.json')
    code = verify_source(source, paths)
    for path, h in lock['baseline_code_sha256'].items(): assert code[path] == h
    assert RECIPE == lock['recipe']
    assert sha(INPUTS/'episodes.json') == lock['episodes_sha256']
    return lock, code


@torch.no_grad()
def extract(output, source, device):
    # Reuse the accepted extractor function, with cached T014 gate and state.
    from .fresh_deadband.select import cross_vectors
    from .clip_signal import FrozenCLIP
    from .learned_prototypes import Prototypes
    from .semantic_ttt import SemanticScorer
    lock, code = preflight(source)
    torch.set_num_threads(1); torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for asset in (lock['clip'], lock['prototypes']): assert sha(Path(asset['path'])) == asset['sha256']
    encoder = FrozenCLIP.from_checkpoint(lock['clip']['path'], device)
    prototypes = Prototypes(torch.load(lock['prototypes']['path'], map_location=device, weights_only=True)['raw'])
    scorer = SemanticScorer(encoder, prototypes)
    output.mkdir(parents=True)
    write(output/'config.json', dict(source_sha=source, source_code_sha256=code,
        lock_sha256=sha(LOCK), feature_schema=lock['feature_schema'], device=device,
        torch_version=torch.__version__, tf32=False, reference_pixels_read=False,
        target_or_mse_read=False, adaptation_rerun=False))
    started = now(); rows = []
    for entry in read(INPUTS/'episodes.json'):
        folder = Path(lock['cache'])/entry['source_directory']
        used = {}
        for method, name in (('semantic','outputs.pt'), (METHOD,'outputs.pt'), (METHOD,'decisions.json')):
            path = folder/method/name; used[method+'/'+name] = sha(path)
            assert used[method+'/'+name] == entry['files'][method][name]['sha256']
        image = torch.load(folder/'semantic/outputs.pt', map_location='cpu', weights_only=True)['identity']['image'].to(device)
        corners = torch.load(folder/METHOD/'outputs.pt', map_location='cpu', weights_only=True)[METHOD]['grid'].to(device)
        decision = read(folder/METHOD/'decisions.json')
        vectors = cross_vectors(image, corners, decision['gate'], lock['gate']['calibration'], scorer)
        rows.append(dict(row_index=entry['row_index'], features=vectors, cache_files_sha256=used,
                         canonical_selected_step=decision['selection']['selected_step']))
        print('Frozen cross features', len(rows), '/120', flush=True)
    assert len(rows) == 120 and all(len(r['features']) == 5 for r in rows)
    write(output/'features.json', rows)
    write(output/'features_frozen.json', dict(started_utc=started, finalized_utc=now(),
        features_sha256=sha(output/'features.json'), config_sha256=sha(output/'config.json'),
        episodes=120, candidate_count=600, reference_pixels_read=False, target_or_mse_read=False))


def feature_inputs(features_dir):
    assert sha(features_dir/'features.json') == read(features_dir/'features_frozen.json')['features_sha256']
    rows = read(features_dir/'features.json')
    # Scatter five unchanged vectors into donor nine-slot layout. The other
    # four slots are never consumed by donor axis_features.
    saved = torch.zeros(120, 9, 28)
    for i, row in enumerate(rows):
        assert row['row_index'] == i
        saved[i, list(CROSS)] = torch.tensor(row['features'])
    return axis_features(saved)


def train(output, features_dir, source):
    lock, code = preflight(source)
    z = feature_inputs(features_dir)
    for name in ('folds.json', 'evaluation.json'):
        assert sha(INPUTS/name) == lock['origins'][name]['sha256']
    raw = (INPUTS/'evaluation.json').read_bytes()  # No full target deserialization.
    folds = read(INPUTS/'folds.json'); entries = read(INPUTS/'episodes.json')
    output.mkdir(parents=True)
    (output/'folds.json').write_bytes((INPUTS/'folds.json').read_bytes())
    write(output/'config.json', dict(task='T020-C', source_sha=source, source_code_sha256=code,
        lock_sha256=sha(LOCK), feature_files_sha256={n:sha(features_dir/n) for n in ('features.json','features_frozen.json','config.json')},
        recipe=RECIPE, target_origin=lock['origins']['evaluation.json'], folds_origin=lock['origins']['folds.json'],
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device='cpu'), development_only=True))
    decisions = []
    for fold in folds:
        for part in ('train', 'heldout'):
            assert [i for i,r in enumerate(entries) if r['image_id'] in fold[part+'_image_ids']] == fold[part]
        targets = training_targets(raw, fold['train'])
        folder = output/f"fold{fold['fold']}"
        decisions += train_fold(z, targets, fold, folder)
        write(folder/'train_targets.json', targets)
        fr = read(folder/'fold_frozen.json')
        for axis in ('x','y'):
            receipt = fr['heads'][axis]
            write(folder/axis/'normalization.json', receipt['normalization'])
            receipt['files_sha256']['normalization.json'] = sha(folder/axis/'normalization.json')
            receipt['train_targets_sha256'] = sha(folder/'train_targets.json')
            write(folder/axis/'receipt.json', receipt)
        fr['finalized_utc'] = now(); write(folder/'fold_frozen.json', fr)
        del targets
        print('Fold', fold['fold'], 'two unchanged heads frozen', flush=True)
    freeze_oof(output, decisions)
    print('OOF frozen', sha(output/'decisions.json'), flush=True)


def summarize(rows):
    groups = {}
    for name in ('nonspatial_pool', *CONDITIONS):
        group = rows if name == 'nonspatial_pool' else [r for r in rows if r['condition'] == name]
        m = {k:statistics.mean(r[k] for r in group) for k in ('H0','H1','H_star')}
        groups[name] = dict(count=len(group), mse=m,
            ratios=dict(H1_over_H0=ratio(m['H1'],m['H0']), H1_over_H_star=ratio(m['H1'],m['H_star'])),
            outcomes=dict(beneficial=sum(r['H1']<r['H0'] for r in group), equal=sum(r['H1']==r['H0'] for r in group), harmful=sum(r['H1']>r['H0'] for r in group)),
            movements={v:sum(r['movement']==v for r in group) for v in ('no_move','x_only','y_only','both')},
            classes={a:{n:sum(r[a+'_class']==c for r in group) for c,n in enumerate(('center','lower','upper'))} for a in ('x','y')},
            agreement={a:dict(count=sum(r[a+'_match'] for r in group),fraction=statistics.mean(r[a+'_match'] for r in group)) for a in ('x','y','joint')})
    clauses={name+'_safety':g['mse']['H1']<=1.01*g['mse']['H0'] for name,g in groups.items()}
    clauses['clean_zero_harmful']=groups['clean']['outcomes']['harmful']==0
    return dict(groups=groups, clauses=clauses, acceptance_vector=list(clauses.values()), passed=sum(clauses.values()),
        verdict='T020-C development OOF '+('positive' if all(clauses.values()) else 'negative'), fresh_qualified=False)


def evaluate(output):
    frozen = read(output/'OOF_frozen.json')
    for name,key in (('decisions','decisions'),('config','config'),('folds','folds')):
        assert sha(output/(name+'.json')) == frozen[key+'_sha256']
    replay = read(output/'head_replay.json'); assert replay['passed'] and replay['decisions_sha256']==frozen['decisions_sha256']
    opened = now(); lock=read(LOCK)
    assert sha(INPUTS/'evaluation.json') == lock['origins']['evaluation.json']['sha256']
    targets = read(INPUTS/'evaluation.json'); rows=[]
    for d,t in zip(read(output/'decisions.json'), targets):
        assert d['row_index']==t['row_index']
        bx,by=d['bx'],d['by']; index=HARD.index((bx,by,0.))
        assert index==d['score_index']
        xmatch=bx==t['bx']; ymatch=by==t['by']
        rows.append(dict(**d,image_id=t['image_id'],condition=t['condition'],H0=t['H0'],H1=t['candidate_mse'][index],H_star=t['H_star'],
            target_bx=t['bx'],target_by=t['by'],x_match=xmatch,y_match=ymatch,joint_match=xmatch and ymatch,
            movement='no_move' if bx==.5 and by==.5 else 'x_only' if by==.5 else 'y_only' if bx==.5 else 'both'))
    report=summarize(rows)
    write(output/'evaluation.json',rows);write(output/'summary.json',report)
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,completed_utc=now(),decisions_sha256=frozen['decisions_sha256'],
        oof_frozen_sha256=sha(output/'OOF_frozen.json'),reference_sha256=sha(INPUTS/'evaluation.json')))
    print(report['verdict'],report['acceptance_vector'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','train','evaluate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--source-sha');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='extract':extract(a.output,a.source_sha,a.device)
    elif a.stage=='train':train(a.output,a.features,a.source_sha)
    else:evaluate(a.output)

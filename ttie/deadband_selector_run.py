"""T019-C: adapt the accepted T018-D final-fit lifecycle to fixed deadband labels."""
import argparse
from pathlib import Path
import platform
import torch
from .direction_probe import CLASSES, RECIPE, fit_head
from .direction_selector import CROSS_NAMES, INFERENCE_SOURCES, HARD, DirectionSelector, cross_features
from .direction_selector_run import replay, SOURCES as FINAL_SOURCES
from .deadband_probe import TARGET, raw_blob, training_targets, SOURCES as OOF_SOURCES
from .energy_local import blob, load_scoring
from .local_geometry import now, sha, write
from .routing.provenance import verify_source

ACCEPTED_B = 'e04a96da31a1a2f359e45ba5f251e04989ab895d'
SOURCES = tuple(dict.fromkeys(FINAL_SOURCES + OOF_SOURCES + ('ttie/deadband_selector_run.py',)))


def training_inputs():
    donor, donor_origin = blob(ACCEPTED_B, 'research_log/T019B_run/config.json')
    scores, hashes = load_scoring(); hashes = {'score_'+k:v for k,v in hashes.items()}
    assert donor['recipe'] == RECIPE and donor['input_artifact_hashes']['targets'] == TARGET
    for key,item in hashes.items(): assert item == donor['input_artifact_hashes'][key]
    for path,expected in donor['source_code_sha256'].items(): assert sha(Path(path)) == expected
    targets = training_targets(raw_blob(TARGET), list(range(120)))
    hashes['targets'] = TARGET; hashes['accepted_b_config'] = donor_origin
    entries = scores['selection']['episodes']; schema = scores['config']['schema']
    assert len(entries) == len(targets) == 120 and schema == donor['feature_schema']
    cross = {name:[r['features'][index] for r in entries] for name,index in zip(CROSS_NAMES,(4,1,7,3,5))}
    return cross, targets, schema, hashes, donor['source_sha']


def train(output, source):
    code = verify_source(source, SOURCES)
    cross, targets, schema, hashes, donor_source = training_inputs()
    output.mkdir(parents=True); started = now()
    runtime = dict(python=platform.python_version(), torch=torch.__version__, device='cpu')
    config = dict(task='T019-C', source_sha=source, source_code_sha256=code, input_artifact_hashes=hashes,
        donor_training_source_sha=donor_source, feature_schema=schema, recipe=RECIPE,
        training_rows=list(range(120)), feature_order=list(CROSS_NAMES), cross_candidate_indices=[4,1,7,3,5],
        runtime=runtime, engineering_freeze_only=True, fresh_qualified=False)
    write(output/'config.json',config); write(output/'replay_features.json',cross)
    z = cross_features(**cross); heads = []; normalization = {}; histories = []
    for axis,features in zip(('x','y'),z):
        labels = torch.tensor([CLASSES.index(targets[i]['b'+axis]) for i in range(120)])
        head,history = fit_head(features,labels); heads.append(head)
        torch.save(dict(state_dict=head.state_dict(),recipe=RECIPE),output/('head_'+axis+'.pt'))
        filename = 'history_'+axis+'.json'; write(output/filename,history); histories.append(filename)
        normalization[axis] = dict(mean=head.x_mean.tolist(),scale=head.x_scale.tolist())
        print('Final',axis,'head trained once:120 rows,100 epochs',flush=True)
    write(output/'normalization.json',normalization)
    write(output/'replay_expected.json',DirectionSelector(*heads).predict(**cross))
    files = ['config.json','head_x.pt','head_y.pt','normalization.json','replay_features.json','replay_expected.json'] + histories
    receipt = dict(task='T019-C',started_utc=started,finalized_utc=now(),source_sha=source,source_code_sha256=code,
        inference_code_sha256={p:code[p] for p in INFERENCE_SOURCES},input_artifact_hashes=hashes,
        donor_training_source_sha=donor_source,runtime=runtime,feature_schema=schema,feature_order=list(CROSS_NAMES),
        feature_construction='concat(f0,fminus-f0,fplus-f0)',candidates=HARD,recipe=RECIPE,normalization=normalization,
        training_rows=120,head_count=2,fit_calls=dict(x=1,y=1),epochs=dict(x=100,y=100),
        files_sha256={f:sha(output/f) for f in files},replay_rows=120,
        replay_role='fixed development features; serialization check only',reference_metrics_opened=False,
        T018E_data_used=False,engineering_freeze_only=True,fresh_qualified=False)
    write(output/'selector_frozen.json',receipt)
    (output/'selector_frozen.sha256').write_text(sha(output/'selector_frozen.json')+'\n',encoding='utf-8')
    print('Selector frozen',sha(output/'selector_frozen.json'),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','replay'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha');p.add_argument('--receipt-sha');a=p.parse_args()
    if a.stage=='train':train(a.output,a.source_sha)
    else:replay(a.output,a.receipt_sha)

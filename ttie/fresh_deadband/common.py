import hashlib
import json
from pathlib import Path
import torch
from ..local_geometry import now, sha, write
from ..routing.provenance import verify_source

LOCK = Path('research_log/T019D_pipeline_lock.json')
CONDITIONS = ('left_right', 'quadrants', 'offset_left_right_40')
CROSS = ((.5,.5,0.), (.4,.5,0.), (.6,.5,0.), (.5,.4,0.), (.5,.6,0.))
HARD = tuple((x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def preparation_bindings(cohort):
    # Byte hashes only for protected manifest/mapping: never decode family/IDs
    # into the deployment process. This is run before its image-read barrier.
    paths=dict(exclusions_sha256=Path('research_log/T019D_exclusions.json'),
        manifest_sha256=cohort/'manifest.json',manifest_frozen_sha256=cohort/'manifest_frozen.json',
        mapping_sha256=cohort/'mapping.json',input_index_sha256=cohort/'inputs/index.json',
        prepared_sha256=cohort/'prepared.json')
    bindings={key:sha(path) for key,path in paths.items()}
    prepared=read(cohort/'prepared.json');mf=read(cohort/'manifest_frozen.json');index=read(cohort/'inputs/index.json')
    assert bindings['mapping_sha256']==prepared['mapping_sha256']
    assert bindings['input_index_sha256']==prepared['inputs_sha256']
    assert bindings['manifest_sha256']==prepared['manifest_sha256']==mf['manifest_sha256']==index['manifest_sha256']
    assert bindings['manifest_frozen_sha256']==index['manifest_frozen_sha256']
    assert bindings['exclusions_sha256']==mf['exclusions_sha256']
    return bindings


def verify_prepared(cohort, config):
    assert 'prepared_sha256' in config, 'Missing pre-inference prepared binding; historical receipts cannot be upgraded retroactively'
    assert sha(cohort/'prepared.json') == config['prepared_sha256'], 'Prepared metadata changed after feature freeze'
    prepared = read(cohort/'prepared.json')
    assert sha(cohort/'mapping.json') == prepared['mapping_sha256'], 'Row mapping differs from frozen preparation'
    assert sha(cohort/'inputs/index.json') == prepared['inputs_sha256'] == config['input_index_sha256']
    assert all(config[key]==value for key,value in preparation_bindings(cohort).items()), 'Preparation chain differs from pre-inference config'
    return prepared


def preflight(source):
    lock = read(LOCK)
    new = [f'ttie/fresh_deadband/{n}.py' for n in ('__init__','common','prepare','select','predict','evaluate')]
    files = list(lock['baseline_code_sha256']) + new + [LOCK.as_posix(), 'research_log/T019D_exclusions.json', 'scripts/run_t019d_a6000.sh', 'research_log/T019D_verify.py']
    code = verify_source(source, files)
    for path, h in lock['baseline_code_sha256'].items(): assert code[path] == h
    selector=Path(lock['selector_dir']);receipt=read(selector/'selector_frozen.json')
    assert sha(selector/'selector_frozen.json')==lock['selector_receipt_sha256']
    for name in ('head_x.pt','head_y.pt'):assert sha(selector/name)==receipt['files_sha256'][name]
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return lock, code

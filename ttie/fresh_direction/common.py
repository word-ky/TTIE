import hashlib
import json
from pathlib import Path
import torch
from ..local_geometry import now, sha, write
from ..routing.provenance import verify_source

LOCK = Path('research_log/T018E_pipeline_lock.json')
CONDITIONS = ('left_right', 'quadrants', 'offset_left_right_40')
CROSS = ((.5,.5,0.), (.4,.5,0.), (.6,.5,0.), (.5,.4,0.), (.5,.6,0.))
HARD = tuple((x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_prepared(cohort, config):
    assert 'prepared_sha256' in config, 'Missing pre-inference prepared binding; historical receipts cannot be upgraded retroactively'
    assert sha(cohort/'prepared.json') == config['prepared_sha256'], 'Prepared metadata changed after feature freeze'
    prepared = read(cohort/'prepared.json')
    assert sha(cohort/'mapping.json') == prepared['mapping_sha256'], 'Row mapping differs from frozen preparation'
    assert sha(cohort/'inputs/index.json') == prepared['inputs_sha256'] == config['input_index_sha256']
    return prepared


def preflight(source):
    lock = read(LOCK)
    new = [f'ttie/fresh_direction/{n}.py' for n in ('__init__','common','prepare','select','predict','evaluate')]
    files = list(lock['baseline_code_sha256']) + new + [LOCK.as_posix(), 'research_log/T018E_exclusions.json', 'scripts/run_t018e_a6000.sh']
    code = verify_source(source, files)
    for path, h in lock['baseline_code_sha256'].items(): assert code[path] == h
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return lock, code

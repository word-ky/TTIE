"""Independent seal verifier: git source bytes and synthetic indices only.

Does not import the producer, metrics, model code, or target-data loaders.
The literal specification digest locks every field, including all endpoint,
sample-policy and information-boundary clauses, against self-consistent reseals.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC_SHA = 'df9c5e4a6c2c5ee8cf812c4535938ecb80d5f6b34fbc659d2fc8302923f93391'
INDEX_SHA = 'f3348c731c348b52eed32160d6b4b2b904101e59a542e1c5dc22850e812da904'
METRIC_COMMIT = '579c3691a80f5b7cadfd706a2fe6750876c53aa0'
SOURCE_HASHES = {
    'research_log/T071B/evaluate.py': '39ae608a0253f506dcd6412b8f4a83b3f966cdb12154c3c6409007865cebc9e7',
    'research_log/T071A/core.py': '45d6b92dee5fb3163da1ea1e9d331a2be36f8991a142bf221167c6d426871858',
    'ttie/ssim_transfer.py': '01d227a8b4caaf8f5705d9c18a5ef685367b6ccf80588d212be92830a04f8556',
}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def git_bytes(ref):
    return subprocess.check_output(['git', 'show', ref], cwd=HERE)

def verify_spec(raw, seal):
    if digest(raw) != SPEC_SHA or seal['analysis_spec_sha256'] != SPEC_SHA:
        raise ValueError('spec differs from independently pinned contract')
    spec = json.loads(raw)
    if seal['bootstrap_indices_sha256'] != INDEX_SHA:
        raise ValueError('bootstrap stream identity mismatch')
    for field in ('inference_runs', 'reference_reads', 'real_metrics'):
        if seal[field] != 0:
            raise ValueError('nonzero execution accounting')
    return spec

def verify():
    spec = verify_spec((HERE/'analysis_spec.json').read_bytes(), json.loads((HERE/'seal.json').read_text()))
    dispatch = json.loads(git_bytes('4c236e84:research_log/T072I/dispatch_manifest.json'))
    core = {k:v for k,v in dispatch.items() if k != 'manifest_sha256'}
    root = digest(json.dumps(core,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
    if root != spec['cohort']['dispatch_root_sha256'] or root != dispatch['manifest_sha256']:
        raise ValueError('accepted dispatch root mismatch')
    names = [row['name'] for row in dispatch['lows']]
    methods = set(spec['methods'])
    jobs = {(row['method'],row['input_name']) for row in dispatch['jobs']}
    if len(names) != 150 or len(set(names)) != 150 or len(dispatch['jobs']) != 450 or jobs != {(m,n) for m in methods for n in names}:
        raise ValueError('canonical coverage mismatch')
    if dispatch['frozen_bindings']['ours']['scientific_source'] != spec['methods']['ours']:
        raise ValueError('Ours source mismatch')
    for method in ('retinexformer','snr_aware'):
        if dispatch['frozen_bindings'][method]['accepted_binding_sha256'] != spec['methods'][method]:
            raise ValueError('baseline binding mismatch')
    bindings = json.loads(git_bytes(METRIC_COMMIT+':research_log/T071B/binding.json'))
    sources = {}
    for path, expected_sha in SOURCE_HASHES.items():
        raw = git_bytes(METRIC_COMMIT+':'+path)
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if digest(raw) != expected_sha or bindings[path] != expected_sha:
            raise ValueError('T071-B runtime source binding mismatch: '+path)
        if blob != spec['metric_source']['blobs'][path]:
            raise ValueError('metric git blob mismatch: '+path)
        sources[path] = {'commit': METRIC_COMMIT, 'blob': blob, 'sha256': expected_sha}
    harness = git_bytes('306975826708bb2a22cc61816ad775d23e108b57:research_log/T072E/verify.py')
    if hashlib.sha1(b'blob '+str(len(harness)).encode()+b'\0'+harness).hexdigest() != spec['information_boundary']['harness_verifier_blob']:
        raise ValueError('accepted harness mismatch')
    # Produce row by row independently of the producer's single matrix call.
    rng = np.random.Generator(np.random.PCG64(20260922))
    indices = np.vstack([rng.integers(0,150,size=150,dtype=np.int64) for _ in range(10000)])
    if digest(indices.astype('<i8').tobytes()) != INDEX_SHA:
        raise ValueError('independent bootstrap reproduction failed')
    return {'classification': 'UHDLL_ANALYSIS_SPEC_SEALED', 'analysis_spec_sha256': SPEC_SHA,
            'dispatch_root_sha256': root, 'canonical_images': len(names), 'canonical_jobs': len(jobs),
            'bootstrap_indices_sha256': INDEX_SHA, 'metric_sources': sources,
            'inference_runs': 0, 'reference_reads': 0, 'real_metrics': 0}

if __name__ == '__main__':
    result = verify()
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

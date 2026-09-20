"""Frozen development calibration -> frozen target-free choices -> offline audit."""
import argparse, csv, hashlib, io, json, math, os, sys, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from PIL import Image
from ttie.common_gain import CommonRegion2
from ttie.ssim_transfer import rgb_ssim
from scripts.evaluate_t026a import independent_ssim, pixels
from research_log.T063A.common import utc, sha, thash, write
from research_log.T065C.core import features,build_bank,predict,base_choose,choose,summarize,FEATURES,RHO,K,confusion
from research_log.T062A.core import losses
from ttie.lolv2_gamma_core import native_rgb

ROOT = Path('/home/wenchang/asdasdsad/wjq/TTIE')
F = Path('/media/wenchang/F/wjq/TTIE')
DEV = F/'runs/T062A-fixed-zr'
TARGET = F/'runs/T063D-fresh-progress'
HERE = Path('research_log/T065C')
NORMALIZATION=Path('research_log/T065A/evidence/model.json')
DEV_LOW=ROOT/'shared/t036a/low'
TARGET_LOW=F/'shared/t063d/low'
DEV_MANIFEST = Path('research_log/T036A_cohort/manifest.json')
TARGET_MANIFEST = Path('research_log/T063D/manifest.json')
DEV_COHORT = '279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
TARGET_COHORT = '3206ea57061f4b45164a81f105f818de6d6d15b342a77797ce9f1eaaccc52554'
DEV_FREEZE = '301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00'
TARGET_FREEZE = '7cdb658d380720f51c2519817953a9e0b6dc0d6f4f2b4bca1deb0bb5cdb3cdd2'


class ReadScope:
    """Record and restrict the actual data reads of the target-free stages."""
    def __init__(self, allowed, out):
        self.allowed = {p.resolve() for p in allowed}; self.out = out.resolve()
        self.enabled = False; self.reads = []
        sys.addaudithook(self.hook)
    def hook(self, event, args):
        if not self.enabled or event != 'open': return
        name, mode, flags = args
        if isinstance(name, int): return
        p = Path(os.fsdecode(name)).resolve()
        if self.out == p or self.out in p.parents: return
        if p in self.allowed:
            self.reads.append(str(p)); return
        if p.suffix in ['.py', '.pyc', '.so']: return
        raise PermissionError('T065-C target-free read boundary: '+str(p))
    def __enter__(self): self.enabled = True; return self
    def __exit__(self, *args): self.enabled = False


def load_inputs(transfer):
    raw, manifest, cohort, freeze = (TARGET, TARGET_MANIFEST, TARGET_COHORT, TARGET_FREEZE) if transfer else (DEV, DEV_MANIFEST, DEV_COHORT, DEV_FREEZE)
    assert sha(manifest) == cohort and sha(raw/'freeze.json') == freeze
    recs = json.loads((raw/'freeze.json').read_bytes())
    assert sha(raw/'config.json') == recs['config_sha256']
    items = json.loads(manifest.read_bytes())['selected']
    assert len(recs['rows']) == len(items) == 100
    return raw, items, recs['rows']


def state_input(raw, rec, transfer):
    d=raw/f"{rec['index']:03d}";lowpath=(TARGET_LOW if transfer else DEV_LOW)/rec['low']
    tracepath=d/('T063_trace.pt' if transfer else 'trace.pt')
    expected=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt']
    assert sha(lowpath)==rec['low_sha256'] and sha(tracepath)==expected
    low=native_rgb(lowpath);trace=torch.load(tracepath,weights_only=True,map_location='cpu')
    if transfer:assert thash(low)==rec['identity_hash']
    return low,trace


def render_features(low,trace):
    x=low.cuda();model=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False)
    images=[];parts=[];totals=[]
    with torch.no_grad():
        for state in trace['states'][:28]:
            model.raw.copy_(state.to(x));y=model(x);components=losses(x,y)
            parts.append(components.cpu().double().tolist())
            totals.append(float(components@components.new_tensor([1.,10.,5.])))
            images.append(y.cpu().clone())
        values=features(x,images,trace['states'][:28],totals).tolist()
    np.testing.assert_allclose(totals,trace['values'][:28],rtol=0,atol=1e-7)
    return images,parts,totals,values


def reconstruct(out, transfer=False, head=None):
    raw, items, recs = load_inputs(transfer)
    allowed=[raw/f'{i:03d}'/('T063_trace.pt' if transfer else 'trace.pt') for i in range(100)]+[(TARGET_LOW if transfer else DEV_LOW)/r['low'] for r in recs]
    scope = ReadScope(allowed, out); rows = []
    with scope:
        for i, (item, rec) in enumerate(zip(items, recs)):
            assert rec['index'] == i and rec['low'] == item['low']
            low, trace = state_input(raw, rec, transfer); images,parts,totals,values=render_features(low,trace)
            hashes = [thash(im) for im in images]
            base_step=base_choose(totals,RHO)
            if transfer: assert base_step==rec['methods']['T063']['selected_step'] and hashes[base_step]==rec['methods']['T063']['output_hash']
            else: assert hashes == rec['rendered_hashes'][:28]
            assert np.isfinite(values).all()
            row = dict(index=i, low=item['low'], components=parts,totals=totals,base_step=base_step,features=values,hashes=hashes)
            if transfer:
                predictions,neighbors=predict(values,head,device='cuda');k=choose(predictions,base_step)
                row.update(probabilities=predictions,neighbors=neighbors,selected_step=k,output_hash=hashes[k])
                with (out/f'{i:03d}.pt').open('xb') as f: torch.save(images[k], f)
                row['output_file_sha256'] = sha(out/f'{i:03d}.pt')
            rows.append(row)
            print('transfer-selected' if transfer else 'development-rendered', i+1, flush=True)
    return rows, scope.reads


def development_rows():
    return [json.loads(Path(f'research_log/T062B/evidence/{i:03d}.json').read_bytes()) for i in range(100)]


def score_transfer(job):
    out, rec, item, ref, independent = job; torch.set_num_threads(1)
    p = Path(out)/f"{rec['index']:03d}.pt"; assert sha(p) == rec['output_file_sha256']
    image = torch.load(p, weights_only=True, map_location='cpu'); assert thash(image) == rec['output_hash']
    original=json.loads((TARGET/'freeze.json').read_bytes())['rows'][rec['index']]
    saved_path=TARGET/f"{rec['index']:03d}"/'outputs.pt';assert sha(saved_path)==original['outputs_sha256']
    saved = torch.load(saved_path, weights_only=True, map_location='cpu')
    for name in ['T026','T036']:assert thash(saved[name])==original['methods'][name]['output_hash']
    opened = utc()
    with zipfile.ZipFile(F/'shared/t022a/LOL-v2.zip') as z: raw = z.read('LOL-v2/Real_captured/'+item['normal'])
    assert hashlib.sha256(raw).hexdigest() == ref['normal_sha256']
    with Image.open(io.BytesIO(raw)) as im: normal = (np.asarray(im.convert('RGB'), dtype=np.float64)/255).astype(np.float32).astype(np.float64)
    metrics = {}
    for name, value in [('selected', image), ('T026', saved['T026']), ('T036', saved['T036'])]:
        x = value[0].permute(1,2,0).numpy().astype(np.float64); d = x-normal
        mse = float(d.ravel() @ d.ravel())/d.size if independent else float(np.mean(d*d))
        metrics[name] = dict(psnr=-10*math.log10(mse), ssim=independent_ssim(x, normal) if independent else rgb_ssim(x, normal))
    return dict(index=rec['index'], low=item['low'], selected_step=rec['selected_step'], metrics=metrics, reference_read_utc=opened)


def metric_rows(scored):
    return [dict(psnr=[r['metrics']['selected']['psnr']], ssim=[r['metrics']['selected']['ssim']],
                 t026_psnr=r['metrics']['T026']['psnr'], t036_psnr=r['metrics']['T036']['psnr'],
                 t036_ssim=r['metrics']['T036']['ssim']) for r in scored]


def main(out):
    torch.set_num_threads(1); torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    binding = json.loads((HERE/'binding.json').read_bytes())
    for p, h in binding.items(): assert sha(p) == h, p
    out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    write(out/'config.json', dict(task='T065-C', source_commit=os.environ['TTIE_SOURCE_COMMIT'], source_binding=binding,
        development_cohort=DEV_COHORT, transfer_cohort=TARGET_COHORT, development_freeze=DEV_FREEZE, transfer_freeze=TARGET_FREEZE,
        gpu=torch.cuda.get_device_name(), physical_gpu=1, torch=torch.__version__, optimizer_runs=0, started_utc=utc()))
    dev, reads = reconstruct(out)
    write(out/'development_features.json', dict(rows=dev, data_reads=reads, completed_utc=utc()))
    quality = development_rows()
    for d, q in zip(dev, quality): assert d['low'] == q['low'] and d['hashes'] == q['state_hashes'][:28]
    training=[]
    for row,q in zip(dev,quality):
        for k in range(28):training.append(dict(index=row['index'],low=row['low'],step=k,features=row['features'][k],target=q['psnr'][k]-q['t026_psnr'],safe=int(q['psnr'][k]-q['t026_psnr']>=-5.614)))
    write(out/'training_table.json',training)
    with (out/'training_table.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['index','step',*FEATURES,'target','safe'])
        for row in training:writer.writerow([row['index'],row['step'],*row['features'],row['target'],row['safe']])
    counts=dict(safe=sum(r['safe'] for r in training),unsafe=sum(1-r['safe'] for r in training),n=len(training));write(out/'class_counts.json',counts)
    normalization=json.loads(NORMALIZATION.read_bytes());head=build_bank(training,normalization);write(out/'bank.json',head)
    outputs=[predict(r['features'],head,exclude_image=r['index'],device='cuda') for r in dev]
    dev_predictions=[v[0] for v in outputs];neighbors=[v[1] for v in outputs]
    steps=[choose(p,r['base_step']) for p,r in zip(dev_predictions,dev)]
    development=dict(probabilities=dev_predictions,neighbors=neighbors,evaluation='leave-one-image-out; all28states of query image excluded',
        selected_steps=steps,base_steps=[r['base_step'] for r in dev],metrics=summarize(quality,steps),
        choices_changed=sum(k!=r['base_step'] for k,r in zip(steps,dev)),confusion=confusion([r['safe'] for r in training],np.asarray(dev_predictions).reshape(-1)),
        selected_step_histogram={str(k):steps.count(k) for k in range(28)},
        base_behavior=[dict(index=r['index'],base_step=r['base_step'],base_probability=p[r['base_step']],selected_step=k,rollback=k<r['base_step']) for r,p,k in zip(dev,dev_predictions,steps)])
    write(out/'development_fit.json',development)
    rule=dict(feature_source='research_log/T065A/core.py',feature_source_sha256=binding['research_log/T065A/core.py'],feature_names=FEATURES,feature_definition='exact 11 features in authorization.md; Y=.299R+.587G+.114B; gradient=mean of concatenated absolute valid H and W forward differences; all raw12 coordinates',
        arithmetic='float32 GPU renderer/objective; float64 GPU image/state features and float64 GPU Euclidean distances; float64 CPU progress and frozen normalization; exact T065-A normalization reused; population standard deviations',
        k=K,rho=RHO,threshold=.5,distance='ordinary Euclidean',weighting='unweighted fraction of5binary labels',tie_order=['distance','development_image_index','step'],rule='keep base if p_safe>=.5; else latest earlier p_safe>=.5; else0',normalization_sha256=sha(NORMALIZATION),class_counts_sha256=sha(out/'class_counts.json'),
        bank_sha256=sha(out/'bank.json'),training_table_sha256=sha(out/'training_table.json'),features_sha256=sha(out/'development_features.json'),development_fit_sha256=sha(out/'development_fit.json'),
        development_cohort=DEV_COHORT,transfer_cohort=TARGET_COHORT,development_freeze=DEV_FREEZE,transfer_input_freeze=TARGET_FREEZE,
        source_binding=binding,source_commit=os.environ['TTIE_SOURCE_COMMIT'],frozen_utc=utc())
    write(out/'selector_manifest.json',rule)
    if not all(development['metrics']['gates'].values()):
        write(out/'result.json',dict(classification='DEVELOPMENT_NEGATIVE',development=development['metrics'],selector_sha256=sha(out/'selector_manifest.json'),completed_utc=utc()));return
    rows,reads=reconstruct(out,True,head)
    write(out/'transfer_freeze.json', dict(rows=rows, data_reads=reads, selector_sha256=sha(out/'selector_manifest.json'),
        cohort_sha256=TARGET_COHORT, input_freeze_sha256=TARGET_FREEZE, selected_outputs=100, completed_utc=utc(), reference_reads=0))
    freeze = json.loads((out/'transfer_freeze.json').read_bytes()); first = utc()
    assert first > freeze['completed_utc'] > rule['frozen_utc']
    write(out/'reference_open.json',dict(first_reference_or_quality_read_utc=first, transfer_freeze_sha256=sha(out/'transfer_freeze.json')))
    for n,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(n)==h,n
    refs=json.loads(Path('research_log/T063D/reference_inputs.json').read_bytes()); items=json.loads(TARGET_MANIFEST.read_bytes())['selected']
    with ProcessPoolExecutor(max_workers=8) as pool: scored=list(pool.map(score_transfer, [(str(out),r,i,n,False) for r,i,n in zip(rows,items,refs)]))
    metrics=summarize(metric_rows(scored),[0]*100)
    result=dict(classification='KNN_SAFETY_GUARD_TRANSFER_PASS' if all(metrics['gates'].values()) else 'TRANSFER_NEGATIVE',
        label='exposed-cohort transfer audit; not fresh qualification', bank_sha256=rule['bank_sha256'], **metrics,
        selector_sha256=sha(out/'selector_manifest.json'), transfer_freeze_sha256=sha(out/'transfer_freeze.json'),
        selected_step_histogram={str(k):sum(r['selected_step']==k for r in rows) for k in range(28)},choices_changed=sum(r['selected_step']!=r['base_step'] for r in rows),
        first_reference_read_utc=first, completed_utc=utc(), seconds=time.perf_counter()-started, optimizer_runs=0)
    tails=[]
    for r in scored:
        if r['index'] in [16,86]:
            n=rows[r['index']]['neighbors'][r['selected_step']]
            tails.append(dict(r,neighbors=[dict(v,safe=head['labels'][v['bank_row']]) for v in n]))
    write(out/'tail_outcomes.json',tails);write(out/'per_image.json',scored); write(out/'result.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())

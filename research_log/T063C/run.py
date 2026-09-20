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
from research_log.T063C.core import losses, progress, choose, calibrate, summarize

ROOT = Path('/home/wenchang/asdasdsad/wjq/TTIE')
F = Path('/media/wenchang/F/wjq/TTIE')
DEV = F/'runs/T062A-fixed-zr'
TARGET = F/'runs/T062CR2-fresh-step27'
HERE = Path('research_log/T063C')
DEV_MANIFEST = Path('research_log/T036A_cohort/manifest.json')
TARGET_MANIFEST = Path('research_log/T062CR1/manifest.json')
DEV_COHORT = '279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
TARGET_COHORT = 'e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2'
DEV_FREEZE = '301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00'
TARGET_FREEZE = '700ae2612234eb139a20c3560b58c85dca1eef3849347c24ffc09575620c07de'


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
        raise PermissionError('T063-C target-free read boundary: '+str(p))
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
    d = raw/f"{rec['index']:03d}"
    if transfer:
        assert sha(d/'outputs.pt') == rec['outputs_sha256']
        assert sha(d/'T062_trace.pt') == rec['methods']['T062']['trace_sha256']
        # This tensor container has images only, no outcome/metric fields. Only identity is used by selector.
        saved = torch.load(d/'outputs.pt', weights_only=True, map_location='cpu')
        trace = torch.load(d/'T062_trace.pt', weights_only=True, map_location='cpu')
        low = saved['identity']; assert thash(low) == rec['identity_hash']
    else:
        for n in ['output.pt', 'trace.pt']: assert sha(d/n) == rec['files'][n]
        saved = torch.load(d/'output.pt', weights_only=True, map_location='cpu')
        trace = torch.load(d/'trace.pt', weights_only=True, map_location='cpu'); low = saved['low']
    return low, trace


def render_components(low, trace):
    x = low.cuda(); model = CommonRegion2(trace['active']).to(x).eval().requires_grad_(False)
    images, components, totals = [], [], []
    with torch.no_grad():
        for state in trace['states'][:28]:
            model.raw.copy_(state.to(x)); y = model(x)
            parts = losses(x, y)
            components.append(parts.cpu().numpy().astype(np.float64).tolist())
            totals.append(float(parts @ parts.new_tensor([1., 10., 5.])))
            images.append(y.cpu().clone())
    assert len(images) == 28
    np.testing.assert_allclose(components, trace['components'][:28].numpy(), rtol=0, atol=1e-7)
    np.testing.assert_allclose(totals, trace['values'][:28], rtol=0, atol=1e-7)
    return images, components, totals


def reconstruct(out, transfer=False, rho=None):
    raw, items, recs = load_inputs(transfer)
    allowed = [raw/f'{i:03d}'/n for i in range(100) for n in (['outputs.pt', 'T062_trace.pt'] if transfer else ['output.pt', 'trace.pt'])]
    scope = ReadScope(allowed, out); rows = []
    with scope:
        for i, (item, rec) in enumerate(zip(items, recs)):
            assert rec['index'] == i and rec['low'] == item['low']
            low, trace = state_input(raw, rec, transfer); images, parts, totals = render_components(low, trace)
            hashes = [thash(im) for im in images]
            if transfer: assert hashes[27] == rec['methods']['T062']['output_hash']
            else: assert hashes == rec['rendered_hashes'][:28]
            best, reduction, q = progress(totals); assert np.isfinite(totals).all()
            row = dict(index=i, low=item['low'], components=parts, totals=totals, best=best, reduction=reduction, progress=q.tolist(), hashes=hashes)
            if transfer:
                k = choose(totals, rho); row.update(selected_step=k, output_hash=hashes[k])
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
    saved = torch.load(TARGET/f"{rec['index']:03d}"/'outputs.pt', weights_only=True, map_location='cpu')
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
    write(out/'config.json', dict(task='T063-C', source_commit=os.environ['TTIE_SOURCE_COMMIT'], source_binding=binding,
        development_cohort=DEV_COHORT, transfer_cohort=TARGET_COHORT, development_freeze=DEV_FREEZE, transfer_freeze=TARGET_FREEZE,
        gpu=torch.cuda.get_device_name(), physical_gpu=1, torch=torch.__version__, optimizer_runs=0, started_utc=utc()))
    dev, reads = reconstruct(out)
    write(out/'development_components.json', dict(rows=dev, data_reads=reads, completed_utc=utc()))
    quality = development_rows()
    for d, q in zip(dev, quality): assert d['low'] == q['low'] and d['hashes'] == q['state_hashes'][:28]
    table, best, passed = calibrate([r['totals'] for r in dev], quality)
    write(out/'threshold_candidates.json', table)
    with (out/'threshold_candidates.csv').open('x', newline='') as f:
        fields=['rho','mean_delta_psnr','median_delta_psnr','regressions_t026','worst_delta_t026','mean_delta_ssim','eligible']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    rule = dict(rule='earliest k in 0..27 with finite L_k <= L_0-rho*(L_0-min L); if D<=1e-12 or none, choose0', statistic='L=spa+10*exp+5*col; q=clip((L_0-L_k)/max(D,1e-12),0,1)',
                arithmetic='accepted float32 GPU component dot product; binary64 progress and threshold', grid='sorted unique finite development q plus0and1',
                rho=best['rho'] if best else None, calibration=best, calibration_pass=passed, candidate_count=len(table),
                development_cohort=DEV_COHORT, transfer_cohort=TARGET_COHORT, source_binding=binding,
                source_commit=os.environ['TTIE_SOURCE_COMMIT'], components_sha256=sha(out/'development_components.json'),
                table_sha256=sha(out/'threshold_candidates.json'), frozen_utc=utc())
    rule['development_selected_steps'] = [choose(r['totals'], best['rho']) for r in dev] if best else []
    write(out/'selector_manifest.json', rule)
    if not passed:
        write(out/'result.json', dict(classification='CALIBRATION_NEGATIVE', calibration=best, selector_sha256=sha(out/'selector_manifest.json'), completed_utc=utc()))
        print('CALIBRATION_NEGATIVE', flush=True); return
    rows, reads = reconstruct(out, True, rule['rho'])
    write(out/'transfer_freeze.json', dict(rows=rows, data_reads=reads, selector_sha256=sha(out/'selector_manifest.json'),
        cohort_sha256=TARGET_COHORT, input_freeze_sha256=TARGET_FREEZE, selected_outputs=100, completed_utc=utc(), reference_reads=0))
    freeze = json.loads((out/'transfer_freeze.json').read_bytes()); first = utc()
    assert first > freeze['completed_utc'] > rule['frozen_utc']
    write(out/'reference_open.json',dict(first_reference_or_quality_read_utc=first, transfer_freeze_sha256=sha(out/'transfer_freeze.json')))
    refs=json.loads(Path('research_log/T062CR2/reference_inputs.json').read_bytes()); items=json.loads(TARGET_MANIFEST.read_bytes())['selected']
    with ProcessPoolExecutor(max_workers=8) as pool: scored=list(pool.map(score_transfer, [(str(out),r,i,n,False) for r,i,n in zip(rows,items,refs)]))
    metrics=summarize(metric_rows(scored),[0]*100)
    result=dict(classification='TARGET_FREE_TRANSFER_PASS' if all(metrics['gates'].values()) else 'TRANSFER_NEGATIVE',
        label='exposed-cohort transfer audit; not fresh qualification', rho=rule['rho'], **metrics,
        selector_sha256=sha(out/'selector_manifest.json'), transfer_freeze_sha256=sha(out/'transfer_freeze.json'),
        selected_step_histogram={str(k):sum(r['selected_step']==k for r in rows) for k in range(28)},
        first_reference_read_utc=first, completed_utc=utc(), seconds=time.perf_counter()-started, optimizer_runs=0)
    write(out/'per_image.json',scored); write(out/'result.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())

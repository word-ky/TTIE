"""Metadata-only cohort freeze, followed by isolated offline corruption synthesis."""
import argparse
from pathlib import Path
import torch
from ..residual_data import evaluation_manifest
from ..natural import load_image, degrade
from .common import CONDITIONS, read, write, sha, now, tensor_sha, preflight


def freeze_manifest(images, output, source):
    lock, code = preflight(source); prior = read('research_log/T020A_exclusions.json')
    excluded = set(prior['excluded_ids'])
    manifest = evaluation_manifest(images, excluded, count=40, split='qualification_t020a')
    assert len({e['image_id'] for e in manifest['images']}) == 40
    assert not {e['image_id'] for e in manifest['images']} & excluded
    assert set(prior['development_T016_T018_ids']) <= excluded
    manifest['selection'] = 'numeric ID ascending; exclude all recorded used/inspected IDs; original min-side>=320; first40'
    manifest['conditions'] = list(CONDITIONS)
    output.mkdir(parents=True)
    write(output/'manifest.json', manifest)
    write(output/'manifest_frozen.json', dict(finalized_utc=now(), manifest_sha256=sha(output/'manifest.json'),
        exclusions_sha256=sha(Path('research_log/T020A_exclusions.json')), source_sha=source, source_code_sha256=code,
        selector_receipt_sha256=lock['selector_receipt_sha256'], source_images=40, episodes=120,
        excluded_count=len(excluded), unique_cohort=True, reference_metric_access=False))
    print('FROZEN cohort', sha(output/'manifest.json'), 'IDs', [e['image_id'] for e in manifest['images']], flush=True)


def synthesize(images, output, source, device):
    _, code = preflight(source); frozen = read(output/'manifest_frozen.json')
    assert frozen['source_sha'] == source and sha(output/'manifest.json') == frozen['manifest_sha256']
    manifest = read(output/'manifest.json'); inputs = output/'inputs'; inputs.mkdir()
    mapping = []; entries = []
    # Source pixels are used only by this offline synthesis process. No metric,
    # model, gate, direction head or label-free adaptation runs here.
    for entry in manifest['images']:
        path = images/entry['filename']; assert sha(path) == entry['sha256']
        clean = load_image(path).to(device)
        for condition in CONDITIONS:
            image = degrade(clean, condition).cpu(); index = len(entries); filename = f'{index:03d}.pt'
            torch.save(dict(image=image), inputs/filename)
            entries.append(dict(row_index=index, file=filename, sha256=sha(inputs/filename), pixels_sha256=tensor_sha(image)))
            mapping.append(dict(row_index=index, image_id=entry['image_id'], filename=entry['filename'], condition=condition, source_sha256=entry['sha256']))
        del clean
    write(output/'mapping.json', mapping)
    write(inputs/'index.json', dict(episodes=entries, manifest_sha256=frozen['manifest_sha256'], manifest_frozen_sha256=sha(output/'manifest_frozen.json')))
    write(output/'prepared.json', dict(completed_utc=now(), source_code_sha256=code, inputs_sha256=sha(inputs/'index.json'),
        mapping_sha256=sha(output/'mapping.json'), manifest_sha256=frozen['manifest_sha256'], episodes=120,
        input_fields=['image'], inference_has_not_run=True, reference_metrics_computed=False))
    print('Prepared 120 degraded-only tensors; no adaptation, direction inference or metrics.', flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['manifest','synthesize'])
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.stage=='manifest':freeze_manifest(a.images,a.output,a.source_sha)
    else:synthesize(a.images,a.output,a.source_sha,a.device)


if __name__=='__main__':main()

"""Original frozen CPU backend; consumes only the opaque GPU feature bundle."""
import argparse
from pathlib import Path
import torch
from ..direction_selector import load_selector, CROSS_NAMES
from .common import preflight, read, write, sha, now, verify_prepared
from .select import install_reference_barrier


def predict(selected, source):
    lock, code=preflight(source)
    cohort=selected.resolve().parent/'cohort'
    verify_prepared(cohort,read(selected/'config.json'))
    blocked=install_reference_barrier(cohort)
    frozen=read(selected/'features_frozen.json')
    assert frozen['source_sha']==source and frozen['episodes']==120
    assert sha(selected/'features.json')==frozen['features_sha256']
    assert sha(selected/'config.json')==frozen['config_sha256']
    rows=read(selected/'features.json');assert len(rows)==120
    assert [r['row_index'] for r in rows]==list(range(120))
    selector=load_selector(lock['selector_dir'],lock['selector_receipt_sha256'])
    batch={name:[r['features'][i] for r in rows] for i,name in enumerate(CROSS_NAMES)}
    predictions=selector.predict(**batch)
    assert not (selected/'decisions_frozen.json').exists()
    write(selected/'decisions.json',[dict(**row,**decision) for row,decision in zip(rows,predictions)])
    assert not blocked
    write(selected/'decisions_frozen.json',dict(started_utc=frozen['started_utc'],finalized_utc=now(),source_sha=source,
        decisions_sha256=sha(selected/'decisions.json'),config_sha256=frozen['config_sha256'],
        case_files_sha256=frozen['case_files_sha256'],features_frozen_sha256=sha(selected/'features_frozen.json'),
        features_sha256=frozen['features_sha256'],source_code_sha256=code,torch_version=torch.__version__,device='cpu',
        batch_size=120,selector_receipt_sha256=lock['selector_receipt_sha256'],manifest_sha256=frozen['manifest_sha256'],
        reference_access=False,blocked_reference_attempts=blocked,frozen_assets_unchanged=True,episodes=120))
    print('ALL 120 DECISIONS FROZEN',sha(selected/'decisions.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selected',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    predict(a.selected,a.source_sha)

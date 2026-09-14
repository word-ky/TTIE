"""Add requested support diagnostics to unchanged accepted metric evaluation."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
f=json.loads((a.audit/'freeze.json').read_bytes());rows=f['rows'];s=json.loads((a.audit/'summary.json').read_bytes())
s.update(task='T032-A',guarded_means='source-only first-exit support rule',changed_selection_count=sum(r['original_step']!=r['selected_step'] for r in rows),
    first_exit_histogram=dict(Counter(str(r['crossing']) for r in rows)),no_exit_fraction=sum(r['crossing'] is None for r in rows)/len(rows),
    step0_exit_fraction=sum(r['crossing']==0 for r in rows)/len(rows),
    selected_support_distance={name:dict(median=float(np.median([r[key] for r in rows])),max=float(max(r[key] for r in rows))) for name,key in [('baseline','original_distance'),('support','selected_distance')]},
    incremental_mean_seconds=float(np.mean([r['guard_seconds'] for r in rows])),
    incremental_percent=100*sum(r['guard_seconds'] for r in rows)/sum(r['trajectory_seconds'] for r in rows))
(a.audit/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s),flush=True)

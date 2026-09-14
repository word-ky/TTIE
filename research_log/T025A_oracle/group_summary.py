"""REFERENCE_ORACLE_ONLY descriptive groups from frozen gate and saved states; no optimization."""
import json,statistics
from pathlib import Path
root=Path('research_log/remote_runs/20260914-075823-ttie-t025a-oracle/artifacts/REFERENCE_ORACLE_ONLY')
rows=json.loads((root/'per_image.json').read_text())['rows']; groups={}; bounds={k:[] for k in ['dark_ev','bright_ev','active_gamma']}
for row in rows:
    state=json.loads((root/f"{row['index']:03d}"/'oracle_states.json').read_text())
    gate=state['gate'];grid=state['physical_grid'][0]
    kinds=[]
    for j,active in enumerate(gate['active']):
        if not active:continue
        kind='dark' if gate['winner'][j]==0 else 'bright';kinds.append(kind)
        bounds[kind+'_ev'].append(grid[0][j//2][j%2]);bounds['active_gamma'].append(grid[1][j//2][j%2])
    name='mixed' if len(set(kinds))==2 else (kinds[0]+'_only' if kinds else 'inactive')
    groups.setdefault(name,[]).append(row)
summary={k:dict(images=len(a),delta_psnr_mean=statistics.mean(r['delta_psnr'] for r in a),delta_psnr_median=statistics.median(r['delta_psnr'] for r in a),delta_ssim_mean=statistics.mean(r['delta_ssim'] for r in a),oracle_psnr_mean=statistics.mean(r['oracle_psnr'] for r in a),ssim_worse=sum(r['delta_ssim']<0 for r in a)) for k,a in groups.items()}
physical={k:dict(count=len(v),minimum=min(v),median=statistics.median(v),maximum=max(v)) for k,v in bounds.items()}
evidence=dict(label='REFERENCE_ORACLE_ONLY',scope='Descriptive frozen-gate grouping only; gate never recomputed or target-relabeled',groups=summary,physical=physical,smallest_psnr_gaps=sorted(rows,key=lambda r:r['delta_psnr'])[:3],largest_psnr_gaps=sorted(rows,key=lambda r:r['delta_psnr'])[-3:])
Path('research_log/T025A_gate_groups.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(dict(groups=summary,physical=physical),indent=2))

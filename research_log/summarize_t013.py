"""Research-lead interim reporting request: distributions from saved T013 data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import torch

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);args=p.parse_args();root=args.audit
alignments=json.loads((root/'alignments.json').read_text());entries=json.loads((root/'artifact_manifest.json').read_text())
def distribution(rows):
    values=[r['cosine'] for r in rows];q=torch.quantile(torch.tensor(values,dtype=torch.float64),torch.tensor([.1,.25,.5,.75,.9],dtype=torch.float64)).tolist()
    return dict(count=len(values),positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),negative=sum(v<0 for v in values),
        positive_fraction=sum(v>0 for v in values)/len(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values),
        quantiles=dict(zip(('p10','p25','p50','p75','p90'),q)),
        bins={'[-1,-0.5)':sum(v<-.5 for v in values),'[-0.5,0)':sum(-.5<=v<0 for v in values),
              '[0,0.5)':sum(0<=v<.5 for v in values),'[0.5,1]':sum(v>=.5 for v in values)})
gradient={'all_active_nonclean':distribution(alignments)}
for c in dict.fromkeys(r['condition'] for r in alignments):gradient[c]=distribution([r for r in alignments if r['condition']==c])
steps={}
for method in ('global_ttt_energy','bilinear2_ttt_energy','region2_ttt_energy'):
    steps[method]={}
    for c in dict.fromkeys(e['condition'] for e in entries):
        decisions=[e['selections'][method] for e in entries if e['condition']==c]
        steps[method][c]=dict(count=len(decisions),histogram=dict(sorted(Counter(d['selected_step'] for d in decisions).items())),
            no_active_bypass=sum(d['bypass']=='no_active' for d in decisions))
result=dict(reference_only=True,method_and_gates_unchanged=True,gradient_cosine=gradient,selected_steps=steps)
(root/'distributions.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# T013 saved-run distributions','',
    'Reporting-only summaries requested in research-lead interim review9a84773. All78raw gradient pairs remain in alignments.json. No additional experiment or selection.','',
    '| Active non-clean group | Count | Positive | Zero | Negative | Mean | Min | P10 | P25 | Median | P75 | P90 | Max |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for c,d in gradient.items():
    values=[d[k] for k in ('count','positive','zero','negative','mean','minimum')]+list(d['quantiles'].values())+[d['maximum']]
    lines.append('| '+c+' | '+' | '.join(f'{v:.9g}' for v in values)+' |')
lines+=['','Overall cosine bins: '+str(gradient['all_active_nonclean']['bins']),
    '', 'Step0 counts include original all-inactive bypasses, which are reported separately. Histograms are step:count.']
for m,conditions in steps.items():
    lines+=['','## '+m,'','| Condition | Selected-step histogram | No-active bypass |','|---|---|---:|']
    for c,d in conditions.items():lines.append('| '+c+' | '+str(d['histogram'])+f" | {d['no_active_bypass']} |")
(root/'distributions.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(overall_gradient=gradient['all_active_nonclean'],primary_steps=steps['region2_ttt_energy']),indent=2))

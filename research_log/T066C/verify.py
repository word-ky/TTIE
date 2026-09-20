"""Independent direct sequence scan and offline classification."""
import argparse,itertools
from pathlib import Path
from research_log.T066C.run import load,sha,write,utc,HERE

def independent_events(rows):
    out=[]
    for i in range(100):
        seq=[r for r in rows if r['index']==i];base=seq[0]['base_step'];prefix=[r['p_safe'] for r in seq if r['step']<=base]
        assert [r['step'] for r in seq]==list(range(28)) and all(r['base_step']==base for r in seq)
        first=None;warnings=[]
        for k,p in enumerate(prefix):
            if p>=.5 and first is None:first=k
            if first is not None and k>first and p<.5:warnings.append(k)
        lengths=[len(list(g)) for value,g in itertools.groupby(range(len(prefix)),key=lambda k:k in warnings) if value]
        out.append(dict(index=i,base_step=base,prefix_probabilities=prefix,first_safe=first,post_safe_unsafe_steps=warnings,first_post_safe_unsafe=min(warnings) if warnings else None,last_post_safe_unsafe=max(warnings) if warnings else None,post_safe_unsafe_count=len(warnings),unsafe_run_count=len(lengths),max_unsafe_run_length=max(lengths,default=0),base_pred_safe=prefix[-1]>=.5,reentry=bool(warnings) and prefix[-1]>=.5))
    return out

def main(out):
    config=load(out/'config.json')
    for p,h in config['bindings'].items():assert sha(p)==h,p
    rows=load('research_log/T066B/evidence/transfer_target_free.json');expected=independent_events(rows);assert expected==load(out/'events.json')
    freeze=load(out/'event_freeze.json');join=load(out/'label_join.json');assert freeze['frozen_utc']<join['first_label_join_utc'] and freeze['label_reads']==0
    assert sha(out/'events.json')==freeze['event_sha256'] and sha('research_log/T066B/evidence/transfer_target_free.json')==freeze['input_sha256'] and sha(out/'event_freeze.json')==join['freeze_sha256']
    for p,h in load(HERE/'evaluation_binding.json').items():assert sha(p)==h,p
    labels=load('research_log/T066B/evidence/transfer_labels.json');result=load(out/'diagnosis.json');counts={k:0 for k in result['base_contingency']};cats={k:0 for k in result['unsafe_state_categories']};unsafe_rows=[];tails=[]
    for i,e in enumerate(expected):
        seq=labels[i*28:(i+1)*28];base=seq[e['base_step']];safe=bool(base['safe']);counts[('safe' if safe else 'unsafe')+('_with_reentry' if e['reentry'] else '_without_reentry')]+=1
        if not safe or i in (16,86):tails.append(dict(e,base_safe=base['safe'],quality_margin=base['quality_margin'],steps_since_last_warning=e['base_step']-max(e['post_safe_unsafe_steps']) if e['post_safe_unsafe_steps'] else None))
        for k,r in enumerate(seq):
            assert r['index']==i and r['step']==k
            if r['safe']:continue
            if k>e['base_step']:cat='outside_prefix'
            elif e['first_safe'] is None:cat='no_first_safe'
            elif k<e['first_safe']:cat='before_first_safe'
            elif e['prefix_probabilities'][k]<.5:cat='during_post_safe_unsafe'
            elif any(p<.5 for p in e['prefix_probabilities'][e['first_safe']:k]):cat='after_safe_reentry'
            else:cat='initial_safe_without_prior_warning'
            cats[cat]+=1;unsafe_rows.append(dict(index=i,step=k,category=cat))
    assert counts==result['base_contingency'] and cats==result['unsafe_state_categories'] and tails==result['tail_rows'] and unsafe_rows==result['unsafe_state_rows']
    assert sum(cats.values())==result['unsafe_state_count']==98
    pair=[r for r in tails if not r['base_safe']];assert [r['index'] for r in pair]==[16,86];num=sum(r['reentry'] for r in pair)
    verdict='PREFIX_REENTRY_SIGNAL_PRESENT' if num==2 else 'PREFIX_REENTRY_SIGNAL_ABSENT' if num==0 else 'PREFIX_REENTRY_SIGNAL_PARTIAL';assert verdict==result['classification']
    write(out/'verification.json',dict(status='PASS',classification=verdict,events_verified=100,unsafe_labels_joined=98,optimizer_runs=0,model_fits=0,verified_utc=utc()));print(verdict,'PASS')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())

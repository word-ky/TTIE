"""Post-run source/selection audit only; no CLIP, rendering, optimization or selection changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import torch
from scipy.stats import spearmanr


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_sha(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--accepted-audit',type=Path,required=True)
    args=parser.parse_args(); base=args.artifacts
    scoring=base/'scoring'; evaluation=base/'evaluation'
    config=read(scoring/'config.json'); receipt=read(scoring/'selection_receipt.json')
    evaluation_receipt=read(evaluation/'evaluation_receipt.json')
    assert sha(scoring/'selection.json')==receipt['selection_sha256']==evaluation_receipt['selection_receipt']['selection_sha256']
    assert sha(scoring/'config.json')==receipt['config_sha256']
    assert receipt['finalized_utc']<evaluation_receipt['reference_opened_utc']
    for path,digest in config['source_code_sha256'].items():
        blob=subprocess.check_output(['git','show',config['source_sha']+':'+path])
        assert hashlib.sha256(blob).hexdigest()==digest==sha(Path(path))
    rows=read(scoring/'selection.json')['episodes']; evaluated=read(evaluation/'evaluation.json')
    summary=read(evaluation/'summary.json'); index=read(base/'label_free_inputs/inputs.json')
    assert len(rows)==len(evaluated)==len(index['episodes'])==120
    assert len({r['episode'] for r in rows})==120
    feature_diff=score_diff=energy_diff=0.; correlated=0
    method='region2_ttt_energy_sobolev'
    for s,e,packed in zip(rows,evaluated,index['episodes']):
        assert s['episode']==e['episode']==packed['episode']
        assert s['input_file_sha256']==packed['sha256']
        original=args.accepted_audit/s['episode']
        outputs=torch.load(original/'outputs.pt',weights_only=True,mmap=True)
        assert tensor_sha(outputs['identity']['image'])==s['pixels_sha256']==packed['pixels_sha256']
        assert tensor_sha(outputs[method]['grid'])==s['corners_sha256']==packed['corners_sha256']
        decision=read(original/method/'decisions.json'); selected=decision['selection']['selected_step']
        assert decision['gate']==s['gate']
        old=torch.load(original/method/'trajectory.pt',weights_only=True,mmap=True)
        feature_diff=max(feature_diff,float((torch.tensor(s['features'][4])-old['features'][selected]).abs().max()))
        score_diff=max(score_diff,float((torch.tensor(s['clip_scores'][4])-old['scores'][selected]).abs().max()))
        energy_diff=max(energy_diff,abs(s['energies'][4]-decision['selection']['scores'][selected]))
        assert all(math.isfinite(v) for v in s['energies'])
        order=sorted(range(9),key=lambda i:(s['energies'][i],i))
        assert s['selected_index']==order[0]==e['selected_index']
        assert s['margin']==s['energies'][order[1]]-s['energies'][order[0]]
        assert s['minimum_ties']==s['energies'].count(min(s['energies']))
        mse=e['reference_mse']; oracle=min(range(9),key=lambda i:(mse[i],i))
        assert e['hard_oracle_index']==oracle and e['selected_mse']==mse[order[0]]
        assert e['hard_oracle_mse']==min(mse) and e['regret']==mse[order[0]]-min(mse)
        if len(set(s['energies']))>1 and len(set(mse))>1:
            corr=float(spearmanr(s['energies'],mse).statistic)
            assert abs(corr-e['spearman'])<1e-12; correlated+=1
        else: assert e['spearman'] is None
    for condition,g in summary['groups'].items():
        group=evaluated if condition=='spatial_pool' else [r for r in evaluated if r['condition']==condition]
        for key in ('selected_mse','region2_mse','hard_oracle_mse','full_oracle_mse','t015_oracle_mse'):
            assert abs(statistics.mean(r[key] for r in group)-g[key])<1e-14
        for key in ('region2','hard_oracle','full_oracle','t015_oracle'):
            assert abs(g['ratios']['selected_over_'+key]-g['selected_mse']/g[key+'_mse'])<1e-14
        assert g['selected_counts']==[sum(r['selected_index']==i for r in group) for i in range(9)]
        assert g['oracle_counts']==[sum(r['hard_oracle_index']==i for r in group) for i in range(9)]
    g=summary['groups']; a=g['spatial_pool']
    clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.05*a['hard_oracle_mse'],
        g['offset_left_right_40']['selected_mse']<=.95*g['offset_left_right_40']['region2_mse'],
        g['left_right']['selected_mse']<=1.01*g['left_right']['region2_mse'],
        g['quadrants']['selected_mse']<=1.01*g['quadrants']['region2_mse']]
    assert clauses==list(summary['clauses'].values()) and sum(clauses)==summary['passed']
    # Report exact measured canonical equivalence; never repair/reselect from references.
    print('Canonical observed differences:', feature_diff, score_diff, energy_diff, flush=True)
    result=dict(passed=True,source_sha=config['source_sha'],scientific_git_blobs=len(config['source_code_sha256']),
        episodes=120,energies=1080,canonical_feature_max_abs=feature_diff,canonical_clip_max_abs=score_diff,
        canonical_energy_max_abs=energy_diff,selection_sha256=receipt['selection_sha256'],
        canonical_bitwise_equal=feature_diff==score_diff==energy_diff==0,
        original_pixels_corners_gates_verified=120,selection_finalized_before_reference=True,
        nonconstant_spearman_verified_with_scipy=correlated,clauses=clauses,
        no_rerender_or_rescoring=True)
    (evaluation/'postrun_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()

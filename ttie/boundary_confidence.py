"""T016-E: fixed confidence fallback on previously frozen OOF score rows."""
import math
import statistics

THRESHOLDS=(0.,.25,.5,.75,1.,1.5,2.,math.inf)
CANONICAL=4


def confidence(scores):
    candidate=min((i for i in range(9) if i!=CANONICAL),key=lambda i:scores[i])
    scale=statistics.pstdev(scores)
    q=0. if scale==0 else (scores[CANONICAL]-scores[candidate])/max(scale,1e-12)
    return candidate,q


def choose(candidate,q,threshold):
    return candidate if q>threshold else CANONICAL


def label(threshold):return 'inf' if math.isinf(threshold) else threshold


def calibrate_fold(oof,fold,reference):
    # Direct reference access is limited to this outer fold's 32 calibration IDs.
    training=[]
    for i in fold['train']:
        row=oof[i]; candidate,q=confidence(row['predictions'])
        training.append((row,candidate,q,reference[row['episode']]['reference_mse']))
    means=[statistics.mean(mse[choose(candidate,q,t)] for _,candidate,q,mse in training) for t in THRESHOLDS]
    selected=min(range(len(THRESHOLDS)),key=lambda i:(means[i],-THRESHOLDS[i]))
    threshold=THRESHOLDS[selected]
    decisions=[]
    for i in fold['heldout']:
        row=oof[i]; candidate,q=confidence(row['predictions'])
        decisions.append(dict(episode=row['episode'],fold=fold['fold'],source_fold=row['fold'],
            predictions=row['predictions'],noncanonical_index=candidate,q=q,threshold=label(threshold),
            selected_index=choose(candidate,q,threshold)))
    calibration=dict(fold=fold['fold'],train_image_ids=fold['train_image_ids'],heldout_image_ids=fold['heldout_image_ids'],
        calibration_episodes=[dict(episode=r['episode'],source_fold=r['fold'],q=q,noncanonical_index=j) for r,j,q,_ in training],
        table=[dict(threshold=label(t),mean_selected_mse=m) for t,m in zip(THRESHOLDS,means)],
        threshold=label(threshold),calibration_count=len(training),heldout_count=len(decisions),
        score_source='previously frozen T016-D OOF; no head loaded or rescored')
    return calibration,decisions

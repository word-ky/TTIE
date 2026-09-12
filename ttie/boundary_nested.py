"""T016-F: isolated outer fold with four fresh inner cross-fit rankers."""
from .boundary_rank_run import fit_fold,sha,write,now
from .boundary_confidence import calibrate_fold


def inner_folds(metadata,outer):
    ids=sorted(outer['train_image_ids']);assignment={image:j%4 for j,image in enumerate(ids)}
    return [dict(fold=k,train=[i for i in outer['train'] if assignment[metadata[i]['image_id']]!=k],
        heldout=[i for i in outer['train'] if assignment[metadata[i]['image_id']]==k],
        train_image_ids=[image for image in ids if assignment[image]!=k],
        heldout_image_ids=[image for image in ids if assignment[image]==k]) for k in range(4)]


def run_outer(x,metadata,episodes,reference,outer,output):
    # This is the only access to the caller's reference mapping: outer-train rows.
    train_reference={episodes[i]:reference[episodes[i]] for i in outer['train']}
    output.mkdir(parents=True)
    inner=inner_folds(metadata,outer);write(output/'inner_folds.json',inner)
    outer_scores,outer_receipt=fit_fold(x,train_reference,episodes,outer,output/'outer_head')
    receipts={'outer_head':outer_receipt};inner_scores=[]
    for fold in inner:
        name=f"inner{fold['fold']}"
        scores,receipt=fit_fold(x,train_reference,episodes,fold,output/name)
        inner_scores.extend(scores);receipts[name]=receipt
    order={e:i for i,e in enumerate(episodes)};inner_scores.sort(key=lambda r:order[r['episode']])
    write(output/'inner_oof.json',inner_scores);write(output/'outer_scores.json',outer_scores)
    score_table=[None]*len(episodes)
    for row in inner_scores+outer_scores:score_table[order[row['episode']]]=row
    calibration,decisions=calibrate_fold(score_table,outer,train_reference)
    calibration['score_source']='fresh inner OOF from four rankers excluding all outer-heldout IDs'
    calibration['inner_oof_sha256']=sha(output/'inner_oof.json')
    write(output/'calibration.json',calibration);write(output/'decisions.json',decisions)
    files=['inner_folds.json','inner_oof.json','outer_scores.json','calibration.json','decisions.json']
    files += [name+'/'+file for name in receipts for file in ('head.pt','history.json','predictions.json','receipt.json')]
    frozen=dict(finalized_utc=now(),outer_fold=outer['fold'],threshold=calibration['threshold'],
        outer_train_image_ids=outer['train_image_ids'],outer_heldout_image_ids=outer['heldout_image_ids'],
        files_sha256={file:sha(output/file) for file in files},head_receipts=receipts,heldout_reference_evaluation=False)
    write(output/'fold_frozen.json',frozen)
    return decisions,frozen

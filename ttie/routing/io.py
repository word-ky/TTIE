"""Reuse T014 traces; persist routing before any evaluation reference access."""
from ..energy_io import run_label_free as run_energy,save_episode as save_energy,hashes
from ..restoration_metrics import evaluate_outputs
from ..stop_io import renamed_row
from ..residual_pilot import write
from .core import BASES,PRIMARY,ORACLE,route


def run_label_free(image,scorer,receipt,head,*,max_steps=40):
    specs={m:(head,m.split('_')[0]) for m in BASES}
    results,old,ts,decisions=run_energy(image,scorer,receipt,None,max_steps=max_steps,energy_specs=specs)
    scores=[decisions[m]['scores'][decisions[m]['selected_step']] for m in BASES]
    routing=route(*scores);chosen=results[routing['selected_basis']]
    results[PRIMARY]=dict(chosen,diagnostics=dict(chosen['diagnostics'],routing=routing))
    decisions['routing']=routing
    return results,old,ts,decisions


def save_episode(directory,results,old,ts,decisions):
    receipt=save_energy(directory,results,old,ts,decisions)
    write(directory/'routing.json',decisions['routing'])
    receipt['episode'].update(hashes(directory,('routing.json',)))
    write(directory/'label_free_receipt.json',receipt)
    return receipt


def evaluate_episode(directory,results,clean,*,image_id,condition):
    rows=evaluate_outputs(results,clean,image_id=image_id,condition=condition)
    by_method={r['method']:r for r in rows}
    chosen=min(BASES,key=lambda m:by_method[m]['mse'])
    oracle=renamed_row(by_method[chosen],ORACLE);rows.append(oracle)
    for r in rows:
        method=chosen if r['method']==ORACLE else r['method']
        if method in (*BASES,PRIMARY):r['selected_step']=results[method]['diagnostics']['selected_step']
    routing=results[PRIMARY]['diagnostics']['routing']
    diagnostic=dict(image_id=image_id,condition=condition,reference_only=True,selected_basis=chosen,
        basis_mse={m:by_method[m]['mse'] for m in BASES},routed_basis=routing['selected_basis'],
        disagreement=chosen!=routing['selected_basis'],routed_mse=by_method[PRIMARY]['mse'],
        oracle_mse=oracle['mse'],mse_regret=by_method[PRIMARY]['mse']-oracle['mse'])
    write(directory/'oracle_diagnostic.json',diagnostic);write(directory/'metrics.json',rows)
    return rows,diagnostic,results[chosen]


def persist_then_evaluate(directory,results,old,ts,decisions,reference,*,image_id,condition):
    files=save_episode(directory,results,old,ts,decisions)
    rows,diagnostic,oracle=evaluate_episode(directory,results,reference(),image_id=image_id,condition=condition)
    return files,rows,diagnostic,oracle

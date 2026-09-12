"""T014 two-head adapters over unchanged label-free T013 trajectories."""
from .energy_io import run_label_free as run_energy,evaluate_episode as evaluate_energy
from .energy_metrics import alignment
from .residual_pilot import write

PRIMARY_METHOD='region2_ttt_energy_sobolev'
CONTROL='region2_ttt_energy_value_only'
ORACLE='oracle_best_sobolev_checkpoint'
METHODS=(CONTROL,'global_ttt_energy_sobolev','bilinear2_ttt_energy_sobolev',PRIMARY_METHOD)


def run_label_free(image,scorer,receipt,heads,*,max_steps=40):
    specs={m:(heads['value_only_control'] if m==CONTROL else heads['sobolev_primary'],m.split('_')[0]) for m in METHODS}
    return run_energy(image,scorer,receipt,None,max_steps=max_steps,energy_specs=specs)


def evaluate_episode(directory,results,trajectories,clean,image,*,image_id,condition,diagnose):
    rows,_,oracle=evaluate_energy(directory,results,trajectories,clean,image,image_id=image_id,condition=condition,
        diagnose=False,primary_method=PRIMARY_METHOD,oracle_method=ORACLE)
    diagnostics=[]
    if diagnose and condition!='clean' and any(trajectories[PRIMARY_METHOD]['gate']['active']):
        diagnostics=[dict(image_id=image_id,condition=condition,method=m,**alignment(image,clean,trajectories[m])) for m in (CONTROL,PRIMARY_METHOD)]
        write(directory/'alignments.json',diagnostics)
    return rows,diagnostics,oracle

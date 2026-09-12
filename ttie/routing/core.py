"""Cross-basis routing sees exactly three frozen selected energy values."""
BASES=('global_ttt_energy_sobolev','bilinear2_ttt_energy_sobolev','region2_ttt_energy_sobolev')
PRIMARY='routed_ttt_energy_sobolev'
ORACLE='oracle_best_basis'


def route(global_score,bilinear_score,region_score):
    scores=(global_score,bilinear_score,region_score)
    order=sorted(range(3),key=lambda i:(scores[i],i))
    return dict(selected_basis=BASES[order[0]],scores=dict(zip(BASES,scores)),
        winner_runner_up_margin=scores[order[1]]-scores[order[0]],tie_order=list(BASES))

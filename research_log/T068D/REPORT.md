# T068-D — DONE / COMPONENT_REGRET_SIGNAL_ABSENT

The exact weighted component-regret score misses the only unsafe T067-C selected endpoint: unsafe above threshold0/1, safe above threshold4/99. The unsafe endpoint ranks52/100 with score0.04199239228904662, below T99_comp0.1910869733048087. Diagnosis only; original choices unchanged.

Source: `2c33a93377d69f83727e847faed20add5ff38e24`; branch: `codex/T068D-component-regret`; authorization: `a9757d99f02500a23a106d6414a2789b8b9b293f`.

Fixed float64 Z=[L_spa,10L_exp,5L_col], zmin/zmax over inclusive [k_FS,k_lambda], a=Z_endpoint-zmin, e=zmax-zmin, R_comp=sum(a)/max(sum(e),1e-12). Only singleton/total-excursion<=1e-12 return0. No clipping/alternative weighting. Fixed lambda=.875, rho=.9857470621423519, threshold.5, accepted T066-A model/features and T062 objective/renderer. No optimizer or model fit.

Target-free development100scores define exact nearest-rank T99_comp=sorted_R[98]=0.1910869733048087; 1 development endpoint exceeds it. Development freeze UTC `2026-09-21T02:43:27.505638+00:00`, SHA256 `6050bb1019a95986fb101a96fc9a507a8bfaf61baaf8ae33fcde1cb64bc233f6`. No development quality/labels opened.

Transfer100row/component/state/output/rule/input freeze UTC `2026-09-21T02:43:28.757377+00:00`, SHA256 `d1cd27066e3e2264ee6022e7ee64943e01adc0478b89381d7cdd7ecb17447453`, reference_reads=0. First transfer quality access boundary UTC `2026-09-21T02:43:28.771208+00:00`; label evidence is hashed/read only afterward. Only existing T067-C endpoint margins are joined; T066-B same-state labels independently crosscheck them.

| Group | Endpoints | Strict R_comp>T99_comp |
|---|---:|---:|
| Unsafe | 1 | 0 |
| Safe | 99 | 4 |

| Index | Safety | k_FS | k_lambda | R_comp | Descending rank | Empirical percentile | Margin vs T026 | Flag |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 3 | safe | 1 | 21 | 0.20692062429118194 | 4 | 0.97 | 3.8173062047006567 | True |
| 55 | safe | 0 | 23 | 0.2070015480496981 | 3 | 0.98 | 2.3689781449980583 | True |
| 68 | safe | 1 | 21 | 0.21789764721946234 | 2 | 0.99 | 4.193759503506262 | True |
| 81 | safe | 0 | 22 | 0.2418496054640402 | 1 | 1.0 | 1.654107992413584 | True |
| 86 | unsafe | 9 | 21 | 0.04199239228904662 | 52 | 0.49 | -6.995482992779127 | False |

Every unsafe endpoint component row (order spa,10exp,5col):

Index86:

| Quantity | Values |
|---|---|
| endpoint_Z | `[0.03851201385259628, 0.6353196501731873, 0.032801711931824684]` |
| zmin | `[0.01436044741421938, 0.6353196501731873, 0.007973843021318316]` |
| zmax | `[0.03851201385259628, 1.752728521823883, 0.032801711931824684]` |
| a | `[0.024151566438376904, 0.0, 0.024827868910506368]` |
| e | `[0.024151566438376904, 1.1174088716506958, 0.024827868910506368]` |
| sum_a | `0.04897943534888327` |
| sum_e | `1.166388306999579` |

Rank1=largest score, exact ties share1+strictly-greater count. Empirical percentile=fraction<=score. Full100rows/ranks and weighted interval histories are in development_freeze/transfer_freeze/joined_endpoints.

Baseline reuse: accepted T068-C endpoint/input/label orchestration, T066-A frozen component history, exact T062A losses. Primary frozen components match trace components exactly. Independent verifier rerenders 4015 interval states from each raw low image, directly recalculates all three losses on GPU using accepted T062A losses, and independently aggregates extrema/regrets in Python float64. Direct loss-component max discrepancy `0.0`; score/component discrepancy `4.440892098500626e-16`; independent T99_comp=0.1910869733048087, error=0.0. Exact flags/labels/diagnosis match. optimizer_runs=0; model_fits=0.

Tests: core3 passed2.10s; local final9 passed16.17s; imports PASS; remote9 passed1.86s. Commands: `python -B -m pytest research_log/T068D research_log/T068C research_log/T067B -q`, `python -B -m research_log.T068D.run --out /media/wenchang/F/wjq/TTIE/runs/T068D-component-regret`, then same with `research_log.T068D.verify`.

Run `20260921-104317-ttie-t068d-component-regret`, exit0 at2026-09-21T10:44:13+08:00; primary 2.1452831529895775 seconds. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068d-component-regret`; output `/media/wenchang/F/wjq/TTIE/runs/T068D-component-regret`. Existing Python3.12/Torch2.4+cu121, physicalGPU1 A6000, TF32off, sequentialMKL, OMP/MKL/OpenBLAS1. 296 source/input bindings, separate two-file evaluation binding post-freeze.

Failures/deviations: none. No new reference image metrics, hypothetical guarded outputs, rollback/stopping selector, tuning, new dataset, fresh/final reference access.

Raw: `/media/wenchang/F/wjq/TTIE/shared/t068d/T068D_raw.tar`, 1648640 bytes, SHA256 `98d30516a2e7b59c81aebfa0be79565d90b89abf8c63b36c92719f781758325a`. Recovery: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t068d/T068D_recovery.tar.gz`, F backup `/media/wenchang/F/wjq/TTIE/shared/t068d/T068D_recovery.tar.gz`, 3332452 bytes, SHA256 `dd714fd206b7dadf43504e9b102ce7a30301c001b303841dcd06f5cb8cf3a36b`; downloaded/hash-verified locally.

Conclusion/next step: this exact component-regret statistic does not isolate the residual unsafe endpoint. Stop at SIGNAL_ABSENT and await the research lead. No guard, component reweighting or alternative aggregation is implemented. Exposed-cohort diagnostic evidence only, not fresh qualification.

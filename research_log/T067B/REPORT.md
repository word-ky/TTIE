# T067-B — DONE / INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN

The fixed robustness-first rule selects strict-interior lambda=0.875. On the original development cohort it passes all five gates: mean/median PSNR delta vs T036 +2.6360346285/+2.3057191323 dB, 1/100 regressions vs T026, worst paired delta -2.4273991686 dB, and mean RGB-SSIM delta +0.0204407487. This is development-only calibration with a frozen in-sample global model, not transfer evidence or qualification. No transfer evaluation was performed.

Authorization `805b340703193c8b244bcf29154d1f41804bfbea`; scientific source `b080b7f3b4f848a0f558c3598b9b713afc1628ac`; branch `codex/T067B-dev-interpolation`. Review only research_log/T067B/** task-owned source/evidence; do not merge unrelated ancestry.

## Exact frozen choices

Only the nine authorized lambdas were evaluated. Endpoints come from the frozen T066-A all-development model probabilities and T063C normalized objective progress. The probabilities were reproduced against the previously frozen T066-B development target-free table; this choice was documented before results in PLAN.md. No LOIO classifier was refitted or substituted. The same frozen final model can be used later if transfer is authorized. This in-sample development calibration should not be confused with T066-A held-out classifier evaluation.

Use T063C float64 clipped progress with denominator floor1e-12, rho=.9857470621423519 and threshold .5. For each image, k_FS is the earliest safe state within k_rho; select earliest k in [k_FS,k_rho] crossing r_FS+lambda*(rho-r_FS). Both endpoint identities were checked for all100images. No fallback, offsets, grid densification or other selector family was used.

All900 candidate choices were bound before development quality access to degraded-image hash, trace hash, selected-state hash and existing frozen rendered-output hash/images-file hash. Existing immutable development output tensors were reused; no optimization was run.

## All nine candidates

| Lambda | Mean delta PSNR | Median delta PSNR | Regressions vs T026 | Worst delta T026 | Mean delta SSIM | All five pass |
|---:|---:|---:|---:|---:|---:|---|
| 0 | -2.7553243375 | -2.4675441866 | 80 | -5.5386899816 | -0.1355450430 | False |
| 0.125 | -2.0997317293 | -1.8372370185 | 74 | -4.6099280133 | -0.0958544716 | False |
| 0.25 | -1.4971035095 | -1.3045454866 | 68 | -4.0077895176 | -0.0654042923 | False |
| 0.375 | -0.8416608348 | -0.7997602360 | 58 | -3.9591408308 | -0.0375180175 | False |
| 0.5 | -0.0869504214 | -0.1924866475 | 31 | -3.8622762616 | -0.0156170853 | False |
| 0.625 | 0.7511393144 | 0.4237759773 | 6 | -3.3413666854 | 0.0035282972 | False |
| 0.75 | 1.6870172622 | 1.3381036032 | 1 | -2.9731603418 | 0.0156396449 | False |
| 0.875 | 2.6360346285 | 2.3057191323 | 1 | -2.4273991686 | 0.0204407487 | True |
| 1 | 3.5695322982 | 3.0938391096 | 3 | -4.0744464137 | 0.0150117752 | True |

Passing candidates ranked by the exact prescribed key: [0.875, 1]. Lambda0.875 has the better worst paired delta; the mean-PSNR and smaller-lambda tie-breakers are available but do not decide this top pair. Every exact selected step and gate value is in candidate_metrics.json.

```json
{
  "classification": "INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN",
  "selected_lambda": 0.875,
  "selected_metrics": {
    "lambda_value": 0.875,
    "selected_steps": [
      23,
      23,
      23,
      22,
      24,
      18,
      21,
      22,
      21,
      22,
      24,
      23,
      23,
      23,
      23,
      21,
      24,
      21,
      21,
      22,
      22,
      24,
      23,
      21,
      23,
      24,
      22,
      24,
      24,
      23,
      24,
      24,
      23,
      21,
      22,
      22,
      16,
      17,
      23,
      20,
      23,
      24,
      23,
      21,
      23,
      24,
      24,
      21,
      19,
      24,
      25,
      23,
      22,
      21,
      9,
      24,
      23,
      17,
      22,
      22,
      23,
      19,
      23,
      23,
      24,
      22,
      22,
      21,
      17,
      24,
      21,
      20,
      21,
      22,
      23,
      22,
      21,
      22,
      22,
      23,
      23,
      24,
      23,
      22,
      22,
      22,
      24,
      24,
      23,
      21,
      23,
      21,
      22,
      22,
      24,
      21,
      21,
      23,
      19,
      23
    ],
    "mean_delta_psnr": 2.636034628456087,
    "median_delta_psnr": 2.3057191322756285,
    "regressions_t026": 1,
    "worst_delta_t026": -2.42739916855475,
    "mean_delta_ssim": 0.02044074873815783,
    "gates": {
      "mean_psnr": true,
      "median_psnr": true,
      "regressions": true,
      "worst": true,
      "mean_ssim": true
    }
  },
  "passing_candidates_ranked": [
    0.875,
    1
  ],
  "selection_key": "maximize worst_delta_t026; exact tie maximize mean_delta_psnr; exact tie minimize lambda",
  "optimizer_runs": 0,
  "model_fits": 0,
  "completed_utc": "2026-09-20T18:43:15.107238+00:00",
  "seconds": 0.7910656160092913
}
```

## Freeze, reference boundary and rule manifest

Candidate freeze SHA-256 `bc22c36e7430fb1b63e9279412877c3b3ad12cde45d3e244652f9a431de2edff`; frozen UTC `2026-09-20T18:43:15.084241+00:00`. The full900row table contains index, low identity/hash, k_FS, k_rho, r_FS, lambda, r_target, selected step/state hash, output hash and images/trace file hashes. Source/input bindings (273) are separate from100development-quality bindings; quality files are first hashed/read after candidate freeze.

```json
{
  "first_development_quality_read_utc": "2026-09-20T18:43:15.095149+00:00",
  "freeze_sha256": "bc22c36e7430fb1b63e9279412877c3b3ad12cde45d3e244652f9a431de2edff"
}
{
  "lambda_value": 0.875,
  "rho": 0.9857470621423519,
  "probability_threshold": 0.5,
  "grid": [
    0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1
  ],
  "probability_model_sha256": "33f878dd090d5e85ba2d099c0e16bc7e4f1731b41f0c5fd90f2a8a5a30f62be0",
  "candidate_freeze_sha256": "bc22c36e7430fb1b63e9279412877c3b3ad12cde45d3e244652f9a431de2edff",
  "candidate_metrics_sha256": "8bcc9febd86d47587532c09a68fc914af76ba16ce962ac1972cccef5fca02251",
  "classification": "INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN",
  "transfer_authorized": false,
  "frozen_utc": "2026-09-20T18:43:15.107950+00:00"
}
```

No exposed transfer reference or target-free transfer table was read. Only original development inputs, frozen model/development probabilities and original development quality were used. No fresh cohort, official test or cross-dataset access occurred. Per-image choices depend only on frozen target-free endpoints and a global scalar; references are used offline after all choices are frozen to select that scalar.

## Independent validation and commands

```json
{
  "status": "PASS",
  "classification": "INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN",
  "selected_lambda": 0.875,
  "candidate_choices": 900,
  "gpu_unique_renders": 900,
  "independent_quality_states": 2800,
  "metric_max_error": 1.0871303857129533e-12,
  "optimizer_runs": 0,
  "model_fits": 0,
  "verified_utc": "2026-09-20T18:43:53.711879+00:00"
}
```

Tests: 7affected passed11.42s;remote7 passed1.47s. Run `20260921-024307-ttie-t067b-interpolation` exited0. Primary calibration elapsed 0.7910656160092913 seconds; independent verification re-rendered900 unique selected states on physical A6000 GPU1 and recomputed all2800 development-state reference PSNR/SSIM on CPU, agreeing with accepted quality evidence to 1.0871303857129533e-12. It independently reconstructs interval selections, state/output hashes, candidate metrics/gates and rule selection. Ranking uses exact frozen metric values after their numerical verification so tiny independent rounding differences cannot manufacture a non-exact tie.

Exact commands are evidence/run/run.sh: affected T063C/T067B pytest, `python -B -m research_log.T067B.run --out /media/wenchang/F/wjq/TTIE/runs/T067B-dev-interpolation`, then the matching `research_log.T067B.verify`, from release `20260921-ttie-t067b-interpolation`. Captured log is evidence/run/train.log. Changes are core.py, run.py, verify.py, test_core.py, PLAN.md, authorization and binding files, plus evidence. No failed run, scientific code change after results, optimizer rerun or model fit occurred. No deviation beyond the explicitly predeclared final-model probability interpretation described above.

## Recovery and next action

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t067b/T067B_raw.tar",
    "bytes": 675840,
    "sha256": "d6189d0936ba981944ef29333c7c83d446b38f1478c23930b89a46d5e5df0b7a"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t067b/T067B_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t067b/T067B_recovery.tar.gz",
    "bytes": 2056071,
    "sha256": "e21c4ec59cc989ff175840de1cbb23d2fe7c36eafa1aeca0916bb5fd059cb6db"
  }
}
```

Archives include complete candidate evidence, source and run receipts; output tensors remain in the original immutable development storage, addressed by hashes in the candidate table. The recovery archive was retrieved locally and its SHA-256 verified. State/report are mirrored to server home/F storage. Freeze lambda0.875 for research-lead review only; no transfer run is authorized in this cycle.

# T066-B — DONE / SELECTED_TAIL_SPECIFIC_FAILURE

The frozen T066-A classifier detects 77/98 unsafe transfer states (78.5714%) across the full trajectories, but misses both unsafe base checkpoints (0/2 detected). The predeclared classification is **SELECTED_TAIL_SPECIFIC_FAILURE**. This diagnostic does not support describing the model as failing on all transfer safety states; its missed tail occurs precisely at the selected checkpoints. No new selector was created or tested.

Authorization `0b766a808324c0af564c4b1b81f3b79a0ea5ea69`; source `e4c011d2bbd183f6227895e6ac3b940aacb6404b`; branch `codex/T066B-support-diagnosis`. Review only research_log/T066B task-owned implementation/evidence; no merge of unrelated historical branch material is requested.

## Frozen implementation and interpretation

Added core.py, run.py, verify.py, test_core.py, PLAN.md, authorization.md, binding.json and evaluation_binding.json. Reused the accepted 19-D T066-A feature tables, exact final coefficients, threshold .5 and all-development normalization (old 11 T065-A means/scales plus final 8 T066-A means/scales). Development geometry excludes all 28 states of the query image while retaining this same frozen all-development normalization, as predeclared in PLAN.md. No LOIO re-normalization or fitting is performed in this diagnostic. Development target-free probabilities use the frozen final model; the historical T066-A 73/81 LOIO classifier recall is separate context.

Feature/model/normalization/source artifacts have 384 SHA-256 bindings. Tables contain only index/step/base-step, frozen target-free features, probability and render hash. Offline transfer labels are written separately after the freeze. Nearest single safe/unsafe development distances use GPU float64 ordinary Euclidean distance; ties within a class use original image-index/step order. A strictly positive margin means closer to safe support. Geometry is diagnostic only and never feeds checkpoint selection.

## State classification

| Scope | States | Safe→safe | Safe→unsafe | Unsafe→safe | Unsafe→unsafe | Unsafe recall |
|---|---:|---:|---:|---:|---:|---:|
| All states | 2800 | 2500 | 202 | 21 | 77 | 77/98 = 0.785714286 |
| k <= k_rho | 2730 | 2439 | 202 | 12 | 77 | 77/89 = 0.865168539 |
| k = k_rho | 100 | 98 | 0 | 2 | 0 | 0/2 = 0 |

Among transfer-unsafe states, 71/98 (72.44898%) are closer to development-safe support. The development LOIO comparison is 61/81 (75.30864%). These geometric fractions alone do not trigger the support-shift category: the authorized first condition (overall transfer unsafe recall < .50) is false. Overall recall >= .50 and false-safe base states determine the single classification above.

## Support distributions and tail rows

Distributions report min, linear-interpolated quartiles, max and mean. Full statewise distances and nearest safe/unsafe bank-row IDs are in evidence/development_geometry.json and evidence/transfer_geometry.json. Bank row = development image index * 28 + state step.

```json
{
  "classification": "SELECTED_TAIL_SPECIFIC_FAILURE",
  "label": "diagnostic-only exposed-cohort audit; no new selector",
  "transfer": {
    "all": {
      "states": 2800,
      "confusion": {
        "safe_pred_safe": 2500,
        "safe_pred_unsafe": 202,
        "unsafe_pred_safe": 21,
        "unsafe_pred_unsafe": 77
      },
      "unsafe_recall": 0.7857142857142857
    },
    "prefix": {
      "states": 2730,
      "confusion": {
        "safe_pred_safe": 2439,
        "safe_pred_unsafe": 202,
        "unsafe_pred_safe": 12,
        "unsafe_pred_unsafe": 77
      },
      "unsafe_recall": 0.8651685393258427
    },
    "base": {
      "states": 100,
      "confusion": {
        "safe_pred_safe": 98,
        "safe_pred_unsafe": 0,
        "unsafe_pred_safe": 2,
        "unsafe_pred_unsafe": 0
      },
      "unsafe_recall": 0.0
    }
  },
  "transfer_unsafe_support": {
    "states": 98,
    "fraction_m_positive": 0.7244897959183674,
    "d_safe": {
      "min": 0.022376604631845133,
      "q25": 0.24960924410870608,
      "median": 0.32402934594755584,
      "q75": 0.9056916170178615,
      "max": 3.2845910281855493,
      "mean": 0.6233226738555318
    },
    "d_unsafe": {
      "min": 0.02116469138459028,
      "q25": 0.3477888323514535,
      "median": 0.4645914415748896,
      "q75": 0.7412346954411599,
      "max": 10.980629348537633,
      "mean": 1.6325064191082903
    },
    "margin": {
      "min": -0.44357963057924765,
      "q25": -0.006022901355019494,
      "median": 0.11554835705377121,
      "q75": 0.23822031598420335,
      "max": 7.696038320352084,
      "mean": 1.0091837452527583
    }
  },
  "development_loio_unsafe_support": {
    "states": 81,
    "fraction_m_positive": 0.7530864197530864,
    "d_safe": {
      "min": 0.060257628819189786,
      "q25": 0.23907585433189854,
      "median": 0.34835052798123034,
      "q75": 0.6614626288320725,
      "max": 15.97493087883408,
      "mean": 0.7843332061266565
    },
    "d_unsafe": {
      "min": 0.11738713547728966,
      "q25": 0.3097580159450124,
      "median": 0.5127947892387559,
      "q75": 0.8020161457631616,
      "max": 23.554513687842654,
      "mean": 1.1418446152501347
    },
    "margin": {
      "min": -0.3190120500569905,
      "q25": 8.81533766474707e-05,
      "median": 0.1133215276179108,
      "q75": 0.24207384709343283,
      "max": 7.579582809008574,
      "mean": 0.35751140912347795
    }
  },
  "unsafe_base_rows": [
    {
      "index": 16,
      "step": 21,
      "p_safe": 0.9999999999999065,
      "safe": 0,
      "quality_margin": -7.1311154303317075,
      "d_safe": 2.116968259553892,
      "d_unsafe": 6.984576534557032,
      "margin": 4.86760827500314,
      "safe_bank_row": 1616,
      "unsafe_bank_row": 907
    },
    {
      "index": 86,
      "step": 25,
      "p_safe": 0.9999999999999065,
      "safe": 0,
      "quality_margin": -10.364944679494553,
      "d_safe": 1.406172879210794,
      "d_unsafe": 7.047229207921262,
      "margin": 5.6410563287104685,
      "safe_bank_row": 2714,
      "unsafe_bank_row": 907
    }
  ],
  "false_safe_base_states": 2,
  "transfer_unsafe_exact_distance_ties": 0,
  "development_classifier_context": "T066-A held-out classifier recall73/81; target-free development table here uses frozen final model",
  "completed_utc": "2026-09-20T14:47:13.653827+00:00",
  "seconds": 17.540653786010807,
  "optimizer_runs": 0,
  "model_fits": 0
}
```

Tail 16, step 21: closest safe development state is image57/step20; closest unsafe is image32/step11. Tail86, step25: closest safe is image96/step26; closest unsafe is image32/step11. Both base states have frozen p_safe=0.9999999999999065 and positive support margins (4.867608275 and 5.641056329). This describes their positions in the fixed space; it does not validate any OOD threshold or replacement guard.

## Ordering and independent verification

```json
{
  "development_sha256": "ea7a58a5e8c585370f5409a8f761536c865a9891197a4be0c8db8a46c98e46aa",
  "transfer_sha256": "e7a9a72e0e844853c87cdae6bed448341afeb11ef0f9da513a4754b8379ced25",
  "model_sha256": "33f878dd090d5e85ba2d099c0e16bc7e4f1731b41f0c5fd90f2a8a5a30f62be0",
  "normalization_sha256": "448bf38b486e93dd5cca8bd2f316b1f5222dd7becc584186fd5781b7fe3c9473",
  "prior_transfer_sha256": "b5d84c5e8945dbaaea5f37a12972c3f6a8c9b81ca542938d7563302a9a66fd79",
  "reference_reads": 0,
  "frozen_utc": "2026-09-20T14:46:56.394410+00:00"
}
{
  "first_reference_quality_read_utc": "2026-09-20T14:46:56.394621+00:00",
  "freeze_sha256": "ab6a239a04993032b5583131fc5a5d0d14db9e8a14f8691762f3965dac770a7f"
}
```

The complete tables were frozen before any transfer reference-quality read in this task. Only then did the primary audit rerender frozen states on A6000 and compute float64 PSNR against authorized normals and exact T026 output. Labels are diagnostic-only; the renderer, objective, optimizer, trajectory, model and checkpoint rule remain unchanged. No new cohort, official test or cross-dataset access occurred.

Independent verification rerendered all 2800 development and 2800 transfer states, recomputed features with independent NumPy formulas, probabilities with an independent logistic expression, transfer PSNR with CPU dot-product MSE, geometry with SciPy cdist, and classification/distributions/masks separately. It checked hashes, render identities, nearest-state IDs, entire-image exclusions, label ordering and all tail rows.

```json
{
  "status": "PASS",
  "classification": "SELECTED_TAIL_SPECIFIC_FAILURE",
  "development_feature_max_error": 7.105427357601002e-15,
  "transfer_feature_max_error": 6.217248937900877e-15,
  "probability_max_error": 1.1179945857975326e-13,
  "metric_max_error": 1.3429257705865894e-12,
  "distance_max_error": 5.329070518200751e-15,
  "development_states": 2800,
  "transfer_states": 2800,
  "model_fits": 0,
  "verified_utc": "2026-09-20T14:49:01.725207+00:00"
}
```

## Tests, commands, failure and recovery

8affected passed8.90s;remote 8 passed in 1.71s. Completed run `20260920-224647-ttie-t066b-support-r1` exited 0. Primary diagnostic elapsed 17.540653786010807 seconds; model fits 0, optimizer runs 0. Rendering and distance computation ran on physical A6000 GPU1; lightweight tests and independent numerical checks used CPU.

Exact shell commands are in evidence/run/run.sh and logs in evidence/run/train.log. The sequence runs the 8 affected pytest tests, `python -B -m research_log.T066B.run --out /media/wenchang/F/wjq/TTIE/runs/T066B-support-diagnosis-r1` and the corresponding `research_log.T066B.verify` from release `20260920-ttie-t066b-support`.

First run `20260920-224522-ttie-t066b-support` exited 1 because the deployment tar omitted `research_log/T063D/reference_inputs.json`. The evaluation hash preflight caught this after the target-free freeze and before transfer labels/reference reads. The exact already-bound Git file was added to the release, and identical committed scientific code reran in a distinct output directory. No scientific code, settings or diagnosis thresholds changed; no quality result motivated the retry. Failed output and logs are preserved in evidence/failed_artifacts and evidence/failed_run. A complete corrected source package is also retained locally at research_log/T066B_source_complete.tar.gz.

## Archives and next step

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t066b/T066B_raw.tar",
    "bytes": 5826560,
    "sha256": "0c7bbd404ec18bc7310e5e9c36bd058dc6e30c4f2ad96c48f78d4370256faa41"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t066b/T066B_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t066b/T066B_recovery.tar.gz",
    "bytes": 5348855,
    "sha256": "ed7df6df45e2be9bc269f47b6ee84840ff3ed2aeed5648f1bbc3518f9a4ba788"
  }
}
```

Recovery archive SHA-256 was verified after local retrieval; source, evidence, successful/failed run receipts are retained locally and on server home/F storage. Recommended next step: research-lead review of checkpoint-specific misses versus overall state recall before authorizing a new selector. This cycle stops with the diagnostic category; no performance rescue, extra model or threshold trial was run.

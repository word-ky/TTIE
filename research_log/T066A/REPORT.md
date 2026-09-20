# T066-A — DONE / TRANSFER_NEGATIVE

The fixed 19-D trajectory-dynamics guard passes development LOIO screening but fails exposed-cohort transfer: 0/100 transfer rollbacks, with worst paired PSNR delta -10.364944679494553 dB. Close this exact guard; await the research lead's next bounded task. This is not fresh qualification or a final benchmark claim.

Authorization: `0eb6086524fc18cb9ffee0b9606fcfc1e1bb5c27`. Scientific source: `c6d7a40a78e35a85befb0d77b3bbd103350f3387`. Branch: `codex/T066A-dynamics-safety`. Only research_log/T066A task-owned changes are relevant; no historical branch merge is requested.

## Implementation and fixed choices

Added core.py, run.py, verify.py, test_core.py, PLAN.md, authorization.md, binding.json and evaluation_binding.json. Evidence includes all 100 fold models/traces, feature tables, labels, probabilities, normalizations, manifests, output hashes and post-freeze metrics.

Exactly 11 unchanged T065-A snapshot features plus the authorized 8 dynamics features. Old 11 normalization is frozen unchanged. As predeclared, each LOIO fold normalizes the new 8 features using only its 99 training images; final normalization uses all 100 development images. Same T065-B class-balanced 19-coefficient logistic Newton convention, lambda .001, unregularized intercept, fixed .5 threshold, 50-update cap and no tuning. Final fit performed once after LOIO passed. No Adam reruns. Rendering/objective and temporal features use A6000 physical GPU1; small float64 model solves run on CPU. Full stored 28-state trajectories are used, so there is no early-stop compute-saving claim.

## Development LOIO

100 image-held-out fits; 2800 predicted states; 2719 safe and 81 unsafe. Confusion: safe→safe 2532, safe→unsafe 187, unsafe→safe 8, unsafe→unsafe 73. Unsafe recall 73/81 = 0.9012345679012346. Development rollbacks 0/100. All five development gates pass.

```json
{
  "mean_delta_psnr": 3.5695322981528443,
  "median_delta_psnr": 3.093839109589621,
  "regressions_t026": 3,
  "worst_delta_t026": -4.074446413712419,
  "mean_delta_ssim": 0.015011775217397242,
  "gates": {
    "mean_psnr": true,
    "median_psnr": true,
    "regressions": true,
    "worst": true,
    "mean_ssim": true
  },
  "eligible": true
}
```

Development selected-step histogram (nonzero): {
  "21": 1,
  "22": 1,
  "23": 2,
  "24": 1,
  "25": 9,
  "26": 21,
  "27": 65
}.

## Exposed transfer audit

```json
{
  "classification": "TRANSFER_NEGATIVE",
  "label": "exposed-cohort transfer audit; not fresh qualification",
  "model_sha256": "33f878dd090d5e85ba2d099c0e16bc7e4f1731b41f0c5fd90f2a8a5a30f62be0",
  "mean_delta_psnr": 3.8619303358477532,
  "median_delta_psnr": 3.814406659510415,
  "regressions_t026": 12,
  "worst_delta_t026": -10.364944679494553,
  "mean_delta_ssim": 0.01842184098158023,
  "gates": {
    "mean_psnr": true,
    "median_psnr": true,
    "regressions": true,
    "worst": false,
    "mean_ssim": true
  },
  "eligible": false,
  "selector_sha256": "2850fa7694dbfd5cf7753461321886d3fbe53c09692586c72dcb1c716af1a907",
  "transfer_freeze_sha256": "b5d84c5e8945dbaaea5f37a12972c3f6a8c9b81ca542938d7563302a9a66fd79",
  "selected_step_histogram": {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 0,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 1,
    "19": 0,
    "20": 0,
    "21": 3,
    "22": 0,
    "23": 0,
    "24": 3,
    "25": 6,
    "26": 22,
    "27": 65
  },
  "choices_changed": 0,
  "first_reference_read_utc": "2026-09-20T13:07:44.577965+00:00",
  "completed_utc": "2026-09-20T13:07:48.362951+00:00",
  "seconds": 159.41020090400707,
  "optimizer_runs": 0
}
```

Tail outcomes and target-free choices:

```json
[
  {
    "index": 16,
    "low": "Train/Low/low00478.png",
    "selected_step": 21,
    "metrics": {
      "selected": {
        "psnr": 15.243922691784427,
        "ssim": 0.730831917478995
      },
      "T026": {
        "psnr": 22.37503812211613,
        "ssim": 0.8275080930423213
      },
      "T036": {
        "psnr": 15.103563283864165,
        "ssim": 0.7611190343853581
      }
    },
    "reference_read_utc": "2026-09-20T13:07:45.344796+00:00",
    "target_free_choice": {
      "base_step": 21,
      "selected_step": 21,
      "output_hash": "079ee64e2f7773c9b952747db0e03c64b4dacd5c32276680d8720cadfbde0b23",
      "p_safe_at_base": 0.9999999999999065
    }
  },
  {
    "index": 86,
    "low": "Train/Low/low00262.png",
    "selected_step": 25,
    "metrics": {
      "selected": {
        "psnr": 13.247833801770003,
        "ssim": 0.6078457175173902
      },
      "T026": {
        "psnr": 23.612778481264556,
        "ssim": 0.7902807908039036
      },
      "T036": {
        "psnr": 13.751995763675106,
        "ssim": 0.6355402917612807
      }
    },
    "reference_read_utc": "2026-09-20T13:07:47.674715+00:00",
    "target_free_choice": {
      "base_step": 25,
      "selected_step": 25,
      "output_hash": "f2018f109f1c13d92cf503af4db7eb05e2852528be8f8b87b44dbead435f755b",
      "p_safe_at_base": 0.9999999999999065
    }
  }
]
```

Rule frozen at 2026-09-20T13:06:43.823547+00:00; all 100 transfer outputs/choices frozen at 2026-09-20T13:07:44.479777+00:00; first reference-quality read at 2026-09-20T13:07:44.577965+00:00. Transfer references enter only post-freeze evaluation. No new cohort, official test or cross-dataset data was accessed.

## Verification and commands

16baseline passed5.81s;3dynamics tests passed4.41s;4including independent passed8.33s;20affected passed8.14s;remote20passed1.64s. Remote run `20260920-210459-ttie-t066a-dynamics` exited 0. The primary run took 159.41020090400707 seconds. Independent verifier rerendered 2800 development and 2800 transfer states, refit all 100 LOIO models and the final model, and checked normalizations, Newton equations, predictions, choices, hashes, ordering and metrics. Final solver executed 50 updates; last step infinity norm 3.3849127746499794e-12 (reporting cap/stop behavior without claiming additional convergence).

```json
{
  "status": "PASS",
  "development_renders": 2800,
  "development_feature_max_error": 7.105427357601002e-15,
  "loio_fits": 100,
  "coefficient_max_error": 1.2807760185751249e-08,
  "normalization_max_error": 1.942890293094024e-16,
  "newton_equation_max_residual": 6.864481205681727e-11,
  "development_metric_max_error": 1.0871303857129533e-12,
  "development_confusion": {
    "safe_pred_safe": 2532,
    "safe_pred_unsafe": 187,
    "unsafe_pred_safe": 8,
    "unsafe_pred_unsafe": 73
  },
  "unsafe_recall": 0.9012345679012346,
  "classification": "TRANSFER_NEGATIVE",
  "transfer_feature_max_error": 6.217248937900877e-15,
  "transfer_renders": 2800,
  "selected_outputs": 100,
  "metric_max_error": 8.846257060213247e-13,
  "gates": {
    "mean_psnr": true,
    "median_psnr": true,
    "regressions": true,
    "worst": false,
    "mean_ssim": true
  },
  "verified_utc": "2026-09-20T13:10:47.351942+00:00"
}
```

Exact executed shell commands are in evidence/run.sh; captured output in evidence/train.log. Scientific commands run the affected pytest suite, research_log/T066A/run.py and research_log/T066A/verify.py from release `20260920-ttie-t066a-dynamics`. No scientific deviations, retries or failures occurred in this run. Archive transfer completed and its SHA-256 matched locally. GitHub push retry after email verification succeeded.

## Durable artifacts

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t066a/T066A_raw.tar",
    "bytes": 306196480,
    "sha256": "4e5bb6ef0b0203dd45cfcce364d3fbe0f4c6fe4b9a5ecc7288c1a5c554884c2a"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t066a/T066A_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t066a/T066A_recovery.tar.gz",
    "bytes": 5101466,
    "sha256": "70feae44ea5fa9dfffd03fcfeddd79e318a98d91579c0a75eb44f53ea4f0efa5"
  }
}
```

Raw archive retains tensors. Recovery archive retains source, JSON/CSV and run receipts; a verified copy is saved locally under research_log/T066A_recovery.tar.gz. No further classifier/threshold/feature changes are authorized in this cycle. The high development state recall did not translate into safer transfer checkpoint choices; both catastrophic tails remain uncorrected. Await lead review.

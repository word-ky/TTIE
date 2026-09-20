# T067-A — DONE / FIRST_SAFE_TRANSFER_NEGATIVE

The exact earliest predicted-safe checkpoint fails all five frozen gates on the exposed cohort. Absolute PSNR/RGB-SSIM is 9.2430042146 / 0.2483033791; mean PSNR delta versus T036 is -2.3669106542 dB, and 75/100 images regress versus exact T026. All100 choices change from k_rho; 31 select step0 and33 step1. Close this exact selector without offsets, persistence, threshold changes or a second trial.

Authorization `a9961040cadd3070ede36601d9d7c9ebeb1f9f05`; scientific source `e20592490094fc43f9a9692a8b7be327c20ecda1`; branch `codex/T067A-first-safe`. Task-owned files are research_log/T067A/**. No historical branch merge is requested.

## Frozen method and implementation

core.py implements only min k<=k_rho with frozen p_safe(k)>=.5, requiring a non-null result and equality with accepted T066-C first_safe. run.py binds low/trace/state identities, renders only the selected frozen 12-D state on A6000 GPU1, saves all100outputs and the complete choice freeze before any reference-quality read. verify.py independently reconstructs earliest crossings, re-renders and verifies all100selected tensors, then computes independent PSNR/RGB-SSIM and five-gate verdict. test_core.py covers exact threshold/earliest selection, step0 and prohibited fallback. PLAN.md and 384 source/input bindings were committed before execution. A pre-deployment follow-up commit includes reused T066-C regression-test dependencies; scientific selector code did not change after testing or results.

The input event/table artifacts are bound to accepted T066-C evidence6daea6cc9280c67d53f6b1939f563d1de274509e and T066-B evidence3291d51cddbdc1b54185c3a905ddab9c3f89e195. The accepted T063B five gates are used unchanged; its calibration-only eligible field is omitted and verdict requires all five. No optimizer rerun, model fit, alternate rule or new data access occurred. Frozen features depend on full trajectory quantities, so this retrospective checkpoint audit does not establish online compute savings.

## Five-gate results

| Quantity | Observed | Gate | Result |
|---|---:|---:|---|
| Mean PSNR delta vs T036 | -2.3669106542 dB | >=2 | FAIL |
| Median PSNR delta vs T036 | -2.8148539978 dB | >0 | FAIL |
| Regressions vs T026 | 75/100 | <=29 | FAIL |
| Worst paired PSNR delta vs T026 | -6.1126334494 dB | >=-5.614 | FAIL |
| Mean RGB-SSIM delta vs T036 | -0.1311716445 | >=-0.001 | FAIL |

```json
{
  "classification": "FIRST_SAFE_TRANSFER_NEGATIVE",
  "label": "exposed-cohort fixed-rule audit only; not fresh qualification",
  "mean_delta_psnr": -2.3669106541634735,
  "median_delta_psnr": -2.814853997847231,
  "regressions_t026": 75,
  "worst_delta_t026": -6.112633449416164,
  "mean_delta_ssim": -0.13117164448636054,
  "gates": {
    "mean_psnr": false,
    "median_psnr": false,
    "regressions": false,
    "worst": false,
    "mean_ssim": false
  },
  "absolute_psnr": 9.243004214561624,
  "absolute_ssim": 0.248303379067755,
  "selected_step_histogram": {
    "0": 31,
    "1": 33,
    "2": 5,
    "3": 8,
    "4": 0,
    "5": 0,
    "6": 5,
    "7": 1,
    "8": 6,
    "9": 3,
    "10": 4,
    "11": 2,
    "12": 2,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 0,
    "21": 0,
    "22": 0,
    "23": 0,
    "24": 0,
    "25": 0,
    "26": 0,
    "27": 0
  },
  "choices_changed": 100,
  "completed_utc": "2026-09-20T17:30:57.730399+00:00",
  "seconds": 6.727627954009222,
  "optimizer_runs": 0,
  "model_fits": 0
}
```

## Exact tails

```json
[
  {
    "index": 16,
    "low": "Train/Low/low00478.png",
    "selected_step": 12,
    "metrics": {
      "selected": {
        "psnr": 25.827903067322794,
        "ssim": 0.8527289730002825
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
    "reference_read_utc": "2026-09-20T17:30:54.359087+00:00",
    "delta_t026": 3.452864945206663,
    "delta_t036": 10.724339783458628
  },
  {
    "index": 86,
    "low": "Train/Low/low00262.png",
    "selected_step": 9,
    "metrics": {
      "selected": {
        "psnr": 19.40289425772575,
        "ssim": 0.7716423601863271
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
    "reference_read_utc": "2026-09-20T17:30:56.899148+00:00",
    "delta_t026": -4.209884223538804,
    "delta_t036": 5.6508984940506455
  }
]
```

Per-image paired metrics and all100selected steps are in evidence/per_image.json and evidence/choice_freeze.json. Tail-specific results cannot override failed cohort gates.

## Freeze and verification

Choice/output freeze SHA-256 `14b4d188f924148b04b50aaf5375be688828efc8fc963e50aed4f6ecc43ab871`; timestamp `2026-09-20T17:30:53.692297+00:00`. The freeze contains index, low path/hash, base step, first-safe step/probability, selected state hash, source trace hash, output tensor hash and saved-file hash. All100outputs were frozen with reference_reads=0. First reference-quality access is recorded below and occurs after this freeze. The reused read scope records only degraded low/trace reads during selection. Reference metadata is first hashed/read after the freeze.

```json
{
  "first_reference_quality_read_utc": "2026-09-20T17:30:53.694203+00:00",
  "freeze_sha256": "14b4d188f924148b04b50aaf5375be688828efc8fc963e50aed4f6ecc43ab871"
}
{
  "status": "PASS",
  "classification": "FIRST_SAFE_TRANSFER_NEGATIVE",
  "selected_states_verified": 100,
  "metric_max_error": 4.334310688136611e-13,
  "gates": {
    "mean_psnr": false,
    "median_psnr": false,
    "regressions": false,
    "worst": false,
    "mean_ssim": false
  },
  "optimizer_runs": 0,
  "model_fits": 0,
  "verified_utc": "2026-09-20T17:31:06.237852+00:00"
}
```

Independent verification recomputes CPU dot-product MSE PSNR and independent separable RGB-SSIM against the exact bound references/T026/T036 outputs. Primary CPU NumPy mean MSE and RGB-SSIM agree within 4.334310688136611e-13. Selected state/render identities and every selected output agree exactly. This remains exposed-cohort method-development evidence, not fresh qualification or a final Ours-vs-baseline claim.

## Tests, receipts, failures and recovery

7affected passed15.96s;remote7 passed1.57s. Run `20260921-013043-ttie-t067a-first-safe` exited0. Primary elapsed 6.727627954009222 seconds. Exact command is evidence/run/run.sh: affected pytest, `python -B -m research_log.T067A.run --out /media/wenchang/F/wjq/TTIE/runs/T067A-first-safe`, then matching `research_log.T067A.verify`, from release `20260921-ttie-t067a-first-safe`. Captured output is evidence/run/train.log. Model/image rendering uses A6000 GPU1; metrics and small checks use CPU. No failed run, binding repair, post-outcome code change or scientific deviation occurred.

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t067a/T067A_raw.tar",
    "bytes": 288512000,
    "sha256": "cddfb038c35c2021c27bc77f2e13ea46df691e79f265268aeae09f50d569268a"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t067a/T067A_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t067a/T067A_recovery.tar.gz",
    "bytes": 1330772,
    "sha256": "a59d657c040dabd2c65234724b84e6912b2b0b6a1dd2663663037ae1d5e9e521"
  }
}
```

Raw archive retains all100selected tensors. Recovery retains source and all non-tensor evidence/receipts; local retrieval SHA-256 matched. State/report are mirrored to server home and F storage. No official test, new Train cohort or cross-dataset set was opened. Recommend closing the exact first-safe rule and returning this negative outcome to the research lead; further rule design is outside this cycle.

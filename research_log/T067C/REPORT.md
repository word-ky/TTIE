# T067-C — DONE / INTERIOR_PROGRESS_TRANSFER_NEGATIVE

The exact development-frozen lambda0.875 interpolation passes four of five exposed-transfer gates, but its worst paired PSNR delta vs T026 is -6.9954829928 dB, below the unchanged -5.614 dB floor. Mean/median gain vs T036 is +2.8848732073/+2.5733132789 dB; 5/100 images regress vs T026; mean RGB-SSIM delta is +0.0291558577. The exact candidate is negative and closed for now. No second lambda or rescue rule was tried.

Authorization `a83b5df3fd1cb0f198a0b0207138c5ed3cf9dd53`; source `21aaceb5e395749fb06d80fd019aafdbc12a9e1f`; branch `codex/T067C-interior-transfer`. Review only research_log/T067C/** task-owned source/evidence; no historical branch merge is requested.

## Exact method and implementation

core.py applies only lambda=.875, rho=.9857470621423519, threshold=.5, and the accepted T063C clipped float64 progress with denominator floor1e-12. The first-safe endpoint and base interval are checked against existing target-free artifacts; earliest target crossing is selected with no fallback. Frozen T066-A final-model probabilities are recomputed and exactly compared with the frozen target-free probability/feature table, and objective history is compared with the immutable trace. The T067B rule manifest and probability-model hash are verified before use.

run.py reuses the accepted T067A GPU rendering/selection-read scope and bound post-freeze reference scoring, but uses the single frozen interval selector. All100selected-state outputs are saved and frozen before opening reference metadata or exact T026/T036 controls. verify.py independently reconstructs probabilities, interval selections, selected-state/output identities, CPU PSNR/RGB-SSIM, gate verdict, histograms and tails. test_core.py compares this fixed rule to T067B on synthetic inputs and checks prohibited fallback. The run never evaluates the other grid candidates. Source includes PLAN.md, authorization, binding manifests and279source/input bindings.

## Five gates and exact result

| Quantity | Observed | Required | Result |
|---|---:|---:|---|
| Mean PSNR delta vs T036 | +2.8848732073 dB | >=2 | PASS |
| Median PSNR delta vs T036 | +2.5733132789 dB | >0 | PASS |
| Regressions vs T026 | 5/100 | <=29 | PASS |
| Worst PSNR delta vs T026 | -6.9954829928 dB | >=-5.614 | FAIL |
| Mean RGB-SSIM delta vs T036 | +0.0291558577 | >=-0.001 | PASS |

```json
{
  "classification": "INTERIOR_PROGRESS_TRANSFER_NEGATIVE",
  "label": "exposed-cohort fixed-rule audit only; not fresh qualification",
  "mean_delta_psnr": 2.884873207346094,
  "median_delta_psnr": 2.5733132788658892,
  "regressions_t026": 5,
  "worst_delta_t026": -6.995482992779127,
  "mean_delta_ssim": 0.029155857730510694,
  "gates": {
    "mean_psnr": true,
    "median_psnr": true,
    "regressions": true,
    "worst": false,
    "mean_ssim": true
  },
  "absolute_psnr": 14.494788076071194,
  "absolute_ssim": 0.4086308812846262,
  "selected_step_histogram": {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 2,
    "9": 1,
    "10": 0,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 1,
    "15": 0,
    "16": 0,
    "17": 1,
    "18": 3,
    "19": 4,
    "20": 11,
    "21": 10,
    "22": 32,
    "23": 23,
    "24": 10,
    "25": 2,
    "26": 0,
    "27": 0
  },
  "choices_changed": 100,
  "completed_utc": "2026-09-20T20:29:28.594578+00:00",
  "seconds": 6.6795929560030345,
  "optimizer_runs": 0,
  "model_fits": 0
}
```

Absolute PSNR/RGB-SSIM is 14.4947880761 / 0.4086308813. All100choices moved earlier than k_rho. This is exposed-cohort method-development evidence only, not fresh qualification or a final benchmark claim. The full-trajectory features also do not establish online early-stop compute savings.

## Post-freeze known-tail diagnosis

```json
[
  {
    "index": 16,
    "low": "Train/Low/low00478.png",
    "selected_step": 19,
    "metrics": {
      "selected": {
        "psnr": 17.47661779313546,
        "ssim": 0.7671811987673736
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
    "reference_read_utc": "2026-09-20T20:29:25.241173+00:00",
    "delta_t026": -4.89842032898067,
    "delta_t036": 2.373054509271295
  },
  {
    "index": 86,
    "low": "Train/Low/low00262.png",
    "selected_step": 21,
    "metrics": {
      "selected": {
        "psnr": 16.61729548848543,
        "ssim": 0.6895647427830512
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
    "reference_read_utc": "2026-09-20T20:29:27.800976+00:00",
    "delta_t026": -6.995482992779127,
    "delta_t036": 2.8652997248103222
  }
]
```

These rows were read only after the complete output freeze. They did not influence lambda, model, interval construction or any exception. Full paired per-image metrics are in evidence/per_image.json.

## Freeze and independent verification

Choice/output freeze SHA-256 `d99fffe009902aa0f4c59691a7c381ded6bb149ca4563673344484e563e5be3e`; timestamp `2026-09-20T20:29:24.549440+00:00`. Freeze records every image/low hash, k_FS, k_rho, r_FS, r_target, selected step/state/trace/output hash, frozen rule/model hashes and reference_reads=0.

Frozen model SHA-256 `33f878dd090d5e85ba2d099c0e16bc7e4f1731b41f0c5fd90f2a8a5a30f62be0`; frozen rule SHA-256 `dc0697e0f3f775fd9ddd46a3bf2c4384cff39ed61dd7693c623b945369334578`. Existing rule is unchanged from accepted T067B; the later authorization permits this single audit.

```json
{
  "first_reference_quality_read_utc": "2026-09-20T20:29:24.551643+00:00",
  "freeze_sha256": "d99fffe009902aa0f4c59691a7c381ded6bb149ca4563673344484e563e5be3e"
}
{
  "status": "PASS",
  "classification": "INTERIOR_PROGRESS_TRANSFER_NEGATIVE",
  "selected_states_verified": 100,
  "metric_max_error": 7.034373084024992e-13,
  "gates": {
    "mean_psnr": true,
    "median_psnr": true,
    "regressions": true,
    "worst": false,
    "mean_ssim": true
  },
  "optimizer_runs": 0,
  "model_fits": 0,
  "verified_utc": "2026-09-20T20:29:36.953861+00:00"
}
```

Primary CPU NumPy mean-MSE PSNR and RGB-SSIM agree with independent CPU dot-product MSE and separable RGB-SSIM to 7.034373084024992e-13; all100state/output identities are independently verified with GPU renders. All test-time reads during selected-output construction are restricted to degraded low images and immutable traces. Evaluation metadata is first hashed/read after freeze.

## Tests, run and recovery

5affected passed15.10s;remote5 passed1.62s. Run `20260921-042913-ttie-t067c-interior` exited0. Primary elapsed 6.6795929560030345 seconds. Selected-state image computation ran on physical A6000 GPU1; lightweight tests/independent metrics used CPU. Exact commands and captured output are evidence/run/run.sh and evidence/run/train.log: affected T067B/T067C pytest, `python -B -m research_log.T067C.run --out /media/wenchang/F/wjq/TTIE/runs/T067C-interior-transfer`, then matching `research_log.T067C.verify`, from release `20260921-ttie-t067c-interior`.

No failed runs, scientific deviations, post-outcome changes, optimizer runs or model fits occurred. No new/fresh cohort, official LOL-v2 Real test or cross-dataset set was opened. No lambda/rho/threshold/window/feature change was attempted.

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t067c/T067C_raw.tar",
    "bytes": 288522240,
    "sha256": "dffdd0c33e9aa88dd174e3c75640191568d00955557ac2716f623c6fef7bbc60"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t067c/T067C_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t067c/T067C_recovery.tar.gz",
    "bytes": 1732091,
    "sha256": "4297cd968b29b2c64645167199d9eed1df727a3121ba564e61442de559b60ad7"
  }
}
```

Raw archive retains all selected tensors; recovery retains complete source, JSON evidence and run receipts. Recovery SHA-256 matched after local retrieval; state/report are mirrored to server home/F storage. Recommended next step: research-lead review of the residual worst-tail failure. Stop this exact transfer candidate without a second selector trial.

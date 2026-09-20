# T066-C — DONE / PREFIX_REENTRY_SIGNAL_ABSENT

Neither catastrophic base state has the predeclared predicted-safe → predicted-unsafe → predicted-safe prefix pattern. Index16 first becomes predicted-safe at step12 and remains safe through base21; index86 first becomes safe at step9 and remains safe through base25. Both have zero post-first-safe unsafe steps/runs and reentry=false. No history-aware selector was implemented or tested.

Authorization `0be01bd7956a84c6f112649ed2fd3a7f42daed5b`. Scientific source `4e4c663827ff0bc7d7b87364c57fd2407ce0bc4d`. Branch `codex/T066C-prefix-reentry`. Prior evidence is exactly `3291d51cddbdc1b54185c3a905ddab9c3f89e195`. Review only research_log/T066C task-owned changes; do not merge historical branch ancestry.

## Aggregate results

| Base label | Without reentry | With reentry |
|---|---:|---:|
| Safe | 94 | 4 |
| Unsafe | 2 | 0 |

All 98 reference-unsafe states are accounted for: 77 before first-safe, 0 during a post-safe unsafe excursion, 0 after predicted-safe reentry, 12 inside the prefix after first-safe without any prior warning, and 9 outside the fixed prefix. No image lacks a first-safe state. The explicit residual/outside-prefix categories were predeclared before the run because the event window ends at k_rho while the requested denominator includes all 28 states. They preserve the exact prefix definition and do not redefine reentry.

The 77 correctly detected unsafe prefix states from T066-B are all before first-safe. Therefore substantial overall/prefix unsafe recall does not establish a warning signal for later quality deterioration after first-safe.

## Exact tail sequences

```json
[
  {
    "index": 16,
    "base_step": 21,
    "prefix_probabilities": [
      0.15575460559169108,
      0.00057071974172511,
      0.00048665821568367573,
      0.006955974299061069,
      0.00561820233953305,
      0.005561909176759197,
      0.006204963082754887,
      0.008688096202189051,
      0.013854688056463525,
      0.0289765641557053,
      0.06785739795698247,
      0.19261475098495764,
      0.5157589516106442,
      0.8414413155835024,
      0.9225793299907852,
      0.991106175916514,
      0.9999973152037972,
      0.9999999999920501,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065
    ],
    "first_safe": 12,
    "post_safe_unsafe_steps": [],
    "first_post_safe_unsafe": null,
    "last_post_safe_unsafe": null,
    "post_safe_unsafe_count": 0,
    "unsafe_run_count": 0,
    "max_unsafe_run_length": 0,
    "base_pred_safe": true,
    "reentry": false,
    "base_safe": 0,
    "quality_margin": -7.1311154303317075,
    "steps_since_last_warning": null
  },
  {
    "index": 86,
    "base_step": 25,
    "prefix_probabilities": [
      0.09498492611325475,
      0.020235995904250743,
      0.013591135954123411,
      0.0786261491797974,
      0.06544269268569197,
      0.0676331708726519,
      0.08419762005071603,
      0.138232451517265,
      0.25422363162576617,
      0.5152647590500631,
      0.8172631926033209,
      0.9592378656541,
      0.9940294428567359,
      0.9994905612479867,
      0.9999876147997797,
      0.9999999487590423,
      0.9999999999506846,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065,
      0.9999999999999065
    ],
    "first_safe": 9,
    "post_safe_unsafe_steps": [],
    "first_post_safe_unsafe": null,
    "last_post_safe_unsafe": null,
    "post_safe_unsafe_count": 0,
    "unsafe_run_count": 0,
    "max_unsafe_run_length": 0,
    "base_pred_safe": true,
    "reentry": false,
    "base_safe": 0,
    "quality_margin": -10.364944679494553,
    "steps_since_last_warning": null
  }
]
```

There is no post-first-safe warning for either tail, so steps_since_last_warning is null. For clarity, the last predicted-unsafe states of any kind were step11 for index16 and step8 for index86 (10 and 17 steps before their bases), respectively; both precede first-safe and cannot satisfy the authorized reentry definition.

## Implementation, ordering and validation

Files: core.py (pure target-free event construction and separate offline diagnosis), run.py, independent verify.py, test_core.py, PLAN.md, authorization.md, binding.json, evaluation_binding.json, plus evidence. Event input is only the committed T066-B target-free table. Threshold .5, frozen probabilities/render identities/base steps are unchanged. Event table stores all requested fields and prefix probabilities. The label artifact is not read, even for hashing, until after the event freeze. Reference label joins never feed event construction.

```json
{
  "event_sha256": "b7b95158ad46c961d64c3d54f24b46075a02784190e33479f66a0fe2a802dff4",
  "input_sha256": "e7a9a72e0e844853c87cdae6bed448341afeb11ef0f9da513a4754b8379ced25",
  "source_commit": "4e4c663827ff0bc7d7b87364c57fd2407ce0bc4d",
  "frozen_utc": "2026-09-20T16:18:39.150904+00:00",
  "label_reads": 0
}
{
  "first_label_join_utc": "2026-09-20T16:18:39.151028+00:00",
  "freeze_sha256": "b43a47de50a080092f5b38a34533146d7a456c70f34d4922606de222ff78c7cd"
}
{
  "status": "PASS",
  "classification": "PREFIX_REENTRY_SIGNAL_ABSENT",
  "events_verified": 100,
  "unsafe_labels_joined": 98,
  "optimizer_runs": 0,
  "model_fits": 0,
  "verified_utc": "2026-09-20T16:18:39.264983+00:00"
}
```

Focused tests: 4 passed0.32s;remote4 passed0.07s. Independent verifier reproduces every event field with a separate sequence/group scan, then independently checks all base contingency counts, all98 unsafe-state assignments, exact tail rows, category, source/input/label hashes and freeze-before-join ordering. Run `20260921-001832-ttie-t066c-reentry` exited0; optimizer_runs=0, model_fits=0. This is lightweight scalar sequence analysis performed on the A6000 server CPU; no model/image computation or GPU experiment was needed.

Exact commands and output are in evidence/run/run.sh and evidence/run/train.log: focused pytest, `python -B -m research_log.T066C.run --out /media/wenchang/F/wjq/TTIE/runs/T066C-prefix-reentry`, then `python -B -m research_log.T066C.verify` with the same output path, from release `20260921-ttie-t066c-reentry`. No failed runs, repairs, scientific deviations, threshold changes, alternate windows, new references, fresh cohorts or held-out data access occurred. Primary timing receipt:

```json
{
  "completed_utc": "2026-09-20T16:18:39.160685+00:00",
  "seconds": 0.0469573139998829,
  "classification": "PREFIX_REENTRY_SIGNAL_ABSENT",
  "optimizer_runs": 0,
  "model_fits": 0
}
```

## Recovery and next step

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t066c/T066C_raw.tar",
    "bytes": 143360,
    "sha256": "4a681201acab66fe35269e5c0d1584032075aa7e6737f049f06722b660fcb9d0"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t066c/T066C_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t066c/T066C_recovery.tar.gz",
    "bytes": 707925,
    "sha256": "d55be975a1c627a6896bc95b9390b334317540d1c340ba9bb34934c57e81a4f4"
  }
}
```

The recovery archive was retrieved and SHA-256 verified locally; source, complete100-image event table, all98 unsafe-state assignments, receipts and verification are retained under project research_log and server home/F storage. The four reentry-positive images have safe bases; neither catastrophic base has reentry. The predeclared signal is absent for both tails. Stop this diagnostic and return to research-lead review; do not infer or implement a different warning/rollback rule in this cycle.

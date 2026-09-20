# T067-D — DONE / INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE

The frozen first-safe→k_rho intervals contain 2465 states; 12 are reference-unsafe across5images. Three unsafe intervals later recover. The two known tails both first become unsafe at step20 and do not recover within the interval. T067-C selects index16 at19 (before its boundary, safe) and index86 at21 (after its boundary, unsafe). This is offline boundary diagnosis only; no cutoff, newlambda or selector was derived or tested.

Authorization `600ebd554d9f26aebc95019ad0ef322f4e2f415f`; source `c5195296580a6be537be9bc37aa8da9bb090707c`; branch `codex/T067D-interval-diagnosis`. Task-owned files: research_log/T067D/** only; no historical branch merge is requested.

## Target-free interval and diagnostic semantics

core.py constructs the exact inclusive interval [k_FS,k_rho] and prescribed q=(r_k-r_FS)/max(rho-r_FS,1e-12), using T063C clipped float64 progress. run.py binds each row to frozen low/trace/state/render identities plus the T067C model/rule hashes and selected-step flag, then freezes the complete table before hashing/reading any quality label. No optimization/refit occurs. Post-freeze diagnosis joins already accepted T066B margins; verifier independently rerenders2800frozenstates on A6000 GPU1 and recomputes CPU reference PSNR/T026 margins.

The inclusive interval can start reference-unsafe despite the classifier's first-safe prediction. Accordingly first_unsafe_step is the first unsafe state in that interval, with first_unsafe_strictly_after_FS also reported to distinguish the strict interpretation. If k_FS is unsafe, the contiguous safe prefix is empty/null. Recovery means any later safe checkpoint after the first unsafe state. These semantics were predeclared in PLAN.md and tested before results.

## Aggregate evidence

```json
{
  "classification": "INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE",
  "images": 100,
  "interval_states": 2465,
  "intervals_with_unsafe": 5,
  "intervals_with_recovery": 3,
  "selected_unsafe": 1,
  "unsafe_interval_states": 12,
  "fixed_q_bins": {
    "[0,.50)": 4,
    "[.50,.75)": 0,
    "[.75,.875)": 1,
    "[.875,1.0001]": 5,
    "outside_fixed_bins": 2
  },
  "outside_fixed_bin_rows": [
    {
      "index": 16,
      "k": 21,
      "q_k": 1.013872898880158
    },
    {
      "index": 86,
      "k": 25,
      "q_k": 1.0082310254735503
    }
  ],
  "optimizer_runs": 0,
  "model_fits": 0,
  "completed_utc": "2026-09-20T20:53:16.517055+00:00",
  "seconds": 1.1563118220074102
}
```

The four requested q bins retain exact boundaries. Two unsafe states exceed the final bin: index16/step21 q=1.0138728989 and index86/step25 q=1.0082310255. This is allowed by the prescribed q formula because k_rho can overshoot rho; q was not clipped. Explicit outside-bin accounting preserves all12unsafe states (4+0+1+5+2), without modifying any bin or dropping states.

## All five intervals containing unsafe states

```json
[
  {
    "index": 16,
    "k_FS": 12,
    "k_rho": 21,
    "selected_step": 19,
    "any_interval_unsafe": true,
    "first_unsafe_step": 20,
    "first_unsafe_strictly_after_FS": 20,
    "first_safe_state_is_reference_safe": true,
    "contiguous_safe_prefix_end": 19,
    "safety_recovers": false,
    "q_first_unsafe": 0.9779425848318375,
    "q_safe_prefix_end": 0.9263225133950844,
    "selected_boundary_relation": "before",
    "selected_margin": -4.89842032898067,
    "selected_safe": true,
    "unsafe_states": 2
  },
  {
    "index": 86,
    "k_FS": 9,
    "k_rho": 25,
    "selected_step": 21,
    "any_interval_unsafe": true,
    "first_unsafe_step": 20,
    "first_unsafe_strictly_after_FS": 20,
    "first_safe_state_is_reference_safe": true,
    "contiguous_safe_prefix_end": 19,
    "safety_recovers": false,
    "q_first_unsafe": 0.8726056856380736,
    "q_safe_prefix_end": 0.8247877630133367,
    "selected_boundary_relation": "after",
    "selected_margin": -6.995482992779127,
    "selected_safe": false,
    "unsafe_states": 6
  },
  {
    "index": 87,
    "k_FS": 0,
    "k_rho": 27,
    "selected_step": 21,
    "any_interval_unsafe": true,
    "first_unsafe_step": 0,
    "first_unsafe_strictly_after_FS": null,
    "first_safe_state_is_reference_safe": false,
    "contiguous_safe_prefix_end": null,
    "safety_recovers": true,
    "q_first_unsafe": 0.0,
    "q_safe_prefix_end": null,
    "selected_boundary_relation": "after",
    "selected_margin": 5.162530604634414,
    "selected_safe": true,
    "unsafe_states": 1
  },
  {
    "index": 88,
    "k_FS": 8,
    "k_rho": 26,
    "selected_step": 22,
    "any_interval_unsafe": true,
    "first_unsafe_step": 8,
    "first_unsafe_strictly_after_FS": null,
    "first_safe_state_is_reference_safe": false,
    "contiguous_safe_prefix_end": null,
    "safety_recovers": true,
    "q_first_unsafe": 0.0,
    "q_safe_prefix_end": null,
    "selected_boundary_relation": "after",
    "selected_margin": 4.268032749631729,
    "selected_safe": true,
    "unsafe_states": 1
  },
  {
    "index": 90,
    "k_FS": 0,
    "k_rho": 26,
    "selected_step": 20,
    "any_interval_unsafe": true,
    "first_unsafe_step": 0,
    "first_unsafe_strictly_after_FS": 1,
    "first_safe_state_is_reference_safe": false,
    "contiguous_safe_prefix_end": null,
    "safety_recovers": true,
    "q_first_unsafe": 0.0,
    "q_safe_prefix_end": null,
    "selected_boundary_relation": "after",
    "selected_margin": 3.313546116700442,
    "selected_safe": true,
    "unsafe_states": 2
  }
]
```

For index16 the contiguous safe prefix ends at19 (q=.9263225134), and unsafe begins at20 (q=.9779425848); selected19 is before the boundary. For index86 the safe prefix also ends at19 (q=.8247877630), and unsafe begins at20 (q=.8726056856); selected21 is after the boundary. Both have no later recovery. These reference-derived boundaries are diagnostic-only and must never become per-image inference inputs or exceptions. Other intervals include recoveries, so the aggregate evidence should retain both behaviors instead of describing all unsafe states as monotone late erosion.

## Freeze and independent checks

Interval freeze SHA-256 `9622106265bd24592f40e5e2ac7f701a9249719453a18fb727a259d96babe3fb`; timestamp `2026-09-20T20:53:16.434112+00:00`; reference_reads=0. Full interval identities, q/r values, model/rule hashes and selected flags are in evidence/interval_freeze.json. Complete100-image boundary evidence is evidence/boundaries.json; joined state margins are evidence/interval_labels.json; known-tail rows are evidence/tail_rows.json.

```json
{
  "first_reference_quality_read_utc": "2026-09-20T20:53:16.464111+00:00",
  "freeze_sha256": "9622106265bd24592f40e5e2ac7f701a9249719453a18fb727a259d96babe3fb"
}
{
  "status": "PASS",
  "classification": "INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE",
  "interval_states_verified": 2465,
  "all_transfer_states_rerendered": 2800,
  "metric_max_error": 1.2363443602225743e-12,
  "optimizer_runs": 0,
  "model_fits": 0,
  "verified_utc": "2026-09-20T20:53:38.714750+00:00"
}
```

All274source/input bindings were checked. Quality labels/reference metadata were first hashed/read only after the interval freeze. Independent verification reconstructs interval steps/q, state hashes, exact render hashes through GPU rerendering, reference margins/safety labels, every boundary/recovery field, selected relation/margin, bins/outside-bin rows and aggregate verdict. Maximum margin discrepancy is 1.2363443602225743e-12.

## Tests, run and artifacts

3focused passed12.51s;remote 3 passed in 1.54s. Run `20260921-045307-ttie-t067d-boundary` exited0. Exact commands are evidence/run/run.sh; output is evidence/run/train.log: focused pytest, `python -B -m research_log.T067D.run --out /media/wenchang/F/wjq/TTIE/runs/T067D-interval-boundary`, then matching `research_log.T067D.verify`, from release `20260921-ttie-t067d-boundary`. Primary diagnosis elapsed 1.1563118220074102 seconds. This timing excludes the independent2800state renderer/metric audit. Model/image verification uses A6000 GPU1; scalar bookkeeping and independent reference metrics use CPU.

No failed runs, post-outcome code changes, scientific tuning, optimizer runs or model fits occurred. No fresh cohort, official LOL-v2 Real test, LSRW/UHD-LL or other dataset was opened. q overshoot and inclusive/strict first-unsafe reporting are predeclared bookkeeping clarifications, not a rule change.

```json
{
  "raw": {
    "path": "/media/wenchang/F/wjq/TTIE/shared/t067d/T067D_raw.tar",
    "bytes": 3850240,
    "sha256": "b6cd462c11f42a14c95d92bbb9f3927bbbffe5dc88668950afe0a7cd80fc1fce"
  },
  "recovery": {
    "path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t067d/T067D_recovery.tar.gz",
    "backup": "/media/wenchang/F/wjq/TTIE/shared/t067d/T067D_recovery.tar.gz",
    "bytes": 1764110,
    "sha256": "0d9f409819d229cfd4be941f5b95fd99e4a62a86fd1cb35429101023f030cc69"
  }
}
```

Recovery archive contains source, complete evidence and receipts and matched SHA-256 after local retrieval. State/report are mirrored to server home/F storage; original trajectories/renders remain in immutable prior storage referenced by hashes. Stop with INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE and await the research lead's next hypothesis. No numerical cutoff or replacement selector is proposed in this cycle.

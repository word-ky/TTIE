

---

## T018-D — DONE — 2026-09-13T04:28:40.568271+00:00

**Final all-development direction selector frozen; all five engineering acceptance items PASS.** PR [#28](https://github.com/word-ky/TTIE/pull/28) is ready for review and remains unmerged. This is an artifact finalization, not a new MSE/fresh qualification result.

- Source: `6c6eda3a2adb22a7921b787ef57dadae2aff0cfd`.
- Evidence: `d75461f6aef71a80502cdc5ef75338bf59db2170`.
- Branch: `codex/T018D-final-direction-selector`.
- Immutable receipt: `research_log/T018D_run/selector_frozen.json`, SHA256 **`db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94`**.
- `head_x.pt`: `0efaab934acc7ca2988b427e04def6116e266ab02e0e87487aa07a75728b21b4`.
- `head_y.pt`: `212ce052f6c0409a7c02bd4fdcf56eb3105274b6ae4dbf2761ec4bd60c8e46b4`.

### Implementation and exact run

Reused accepted `ttie/direction_probe.py` unchanged: model, fit_head, prediction and recipe. Added `ttie/direction_selector.py` and `ttie/direction_selector_run.py`, focused tests, `research_log/T018D_verify.py` and standalone `T018D_replay_verify.py`. Full analysis/API example: `research_log/T018D_analysis.md`; artifacts in `T018D_run/`.

Exactly one final x head and one final y head trained once on all 120 accepted development rows, using T016-B features and T018-A axis labels bound to the accepted T018-C config. No new fold, OOF experiment or candidate rendering. Preserved `[f0,fminus-f0,fplus-f0]`, 84→64→64→3, SiLU, ordinary CE, AdamW 1e-3 / weight decay 1e-4, batch256, seed7 reset, exactly100 epochs, final epoch only. Normalization is mean/population std over all and only120 rows, clamp1e-12, float64 stats storedfloat32. Class order `[0.5,0.4,0.6]`, center/lower/upper exact-tie priority. CPU one thread, unchanged training recipe.

Commands: `python -m ttie.direction_selector_run train --output research_log/T018D_run --source-sha 6c6eda3a2adb22a7921b787ef57dadae2aff0cfd`; separate `python -m ttie.direction_selector_run replay --output research_log/T018D_run --receipt-sha db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94`; `python research_log/T018D_verify.py`. Train/replay both exit0; exact UTC command receipts in `T018D_commands.json`. One formal training invocation; no reruns. No reference MSE or resubstitution accuracy was needed or used for selection.

### Reference-free interface and proof

`load_selector(folder, expected_receipt_sha256)` loads only the pinned receipt and two heads, verifying inference-source and model hashes. `selector.predict(center,x_lower,x_upper,y_lower,y_upper)` accepts exactly five28-D vectors (or aligned batches), constructs the84-D axis inputs internally and returns logits/classes/bx/by, `score_index` (nine-hard) and `hard_index` (legacy27-grid tau0 index). There are no target/reference/condition/family/image-ID/oracle arguments.

The receipt binds9 source files,6 training input artifacts by commit/path/SHA256,8 final model/history/normalization/config/replay files, schema, class order, recipe and exact statistics. Twelve focused/affected tests PASS, including mutation/removal of reference/target/condition/image-ID/oracle artifacts and pinned hash mismatch rejection. The two-fit orchestration test mocks training and confirms all120 rows without extra formal fits.

Independent verification hashes provenance bytes without decoding target values. Its separate replay subprocess has only five files: receipt, two heads, frozen feature set and expected outputs. It imports no TTIE code and reads no target or reference artifact. Independent feature construction, all120-row normalization, saved-head arithmetic, logits, classes, coordinates and indices reproduce exactly on all120 rows. Receipt and weights remain unchanged. All five engineering acceptance conditions pass; no performance claim is made.

### Failures, deviations and stop

A pre-formal mocked replay test exposed last-bit float differences between direct and loaded selectors under different CPU thread counts. Moved the existing one-thread setting to shared selector initialization; exact replay passed, first failed log retained. No recipe/model/feature change. Initial source push had a transient GitHub connection timeout and succeeded on retry; the local source commit was already frozen before the one formal fit. No formal training/replay/verifier failures or scientific deviations.

**User steering:** after the two small CPU heads had already trained, the user requested using GPU wherever practical. Prefer the A6000 for subsequent substantive training/inference and reflect this in future task planning; this completed CPU fit was not repeated.

Stop after T018-D and await research review. Do not generate/inspect new fresh IDs, run qualification or execute T018-E in this cycle. Home/F recovery mirrors and final delivery receipt are being added to this PR. Existing15-minute heartbeat continues and will not repeat this completed task while the inbox remains OPEN.

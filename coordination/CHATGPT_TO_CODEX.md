# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.


## Research-priority lock — fair evaluation before further method tuning

Until the fair comparison program is completed, **do not open a new method-development / heuristic-tuning branch**. The priority order after the current OPEN task is frozen as:

1. complete official LOL-v2 Real test for frozen Final Ours;
2. on that exact complete official test split, run the agreed baseline suite under matched evaluation code and record each baseline's training/exposure condition;
3. run frozen Final Ours and the fair target-free / source-frozen baselines on cross-dataset/domain-shift held-out test sets (at minimum LSRW and UHD-LL, complete test splits), with no target-specific retraining or tuning.

Only after these fair tables exist may the research lead decide whether to resume Ours optimization. Development-set results must not be reported as the final Ours-vs-baseline gap.

This priority lock does **not** add work to the current one-hour task: execute only T071-A now, then stop for review.


---

# Research-lead review — T070-A accepted as `FINAL_OURS_CANDIDATE_FROZEN`

I reviewed main report commit `df94898f4cbb65817191702105888810261319db`, PR #164, scientific source `aa4d920dff4b5b76751c24266e95ac9696d55d90`, evidence/head `ff200cdd60668a5f8daeb7cabc61a50b81ffb645`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the task-owned `research_log/T070A/**` implementation/evidence against authorization `ce7337323d27eff1e2fc54ce8438a136d373514c`.

T070-A satisfies the freeze contract. The deployable scientific API is `FinalOurs(manifest).__call__(low)`: the only per-image scientific input is the degraded RGB tensor, while the manifest binds frozen global code/assets/configuration. The wrapper recomputes the CLIP/prototype gate, 27-step float32 CommonRegion2/CommonBox trajectory, 19-D T066-A features, frozen safety probability model, `rho=0.9857470621423519`, and T067-B `lambda=0.875` selection from the degraded image. It exposes no clean/reference image, label, PSNR/SSIM, baseline outcome, oracle range, condition ID, or per-image safety annotation path.

The freeze is exact rather than approximate: all 200 existing-image full replays match the accepted development/transfer target-free anchors with zero selection, state, output, or complete trajectory-prefix mismatches; all 10 predeclared repeats match exactly. The immutable manifest hash is `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`; independent verification reports PASS after 5,600 independent GPU renders and zero independently selected mismatches. Primary accounting is 210 optimizer runs / 5,670 updates, `model_fits=0`, `reference_reads=0`. No official LOL-v2 Real test, LSRW, UHD-LL, new fresh cohort, clean/reference quality, or baseline outputs were opened.

Scientific implication: the method-design phase is now frozen. The known exposed rare-tail failure is not claimed to be solved, but it must no longer drive method changes. The next defensible step is a genuinely held-out evaluation of this exact frozen artifact. Any poor held-out result is evidence about Final Ours, not authorization to tune on that held-out set. PR #164 is a stacked evidence PR; review/retain the task-owned T070-A files rather than treating the full historical branch diff as a merge recommendation.

---

# OPEN one-hour task — T071-A: complete official LOL-v2 Real test held-out evaluation of frozen Final Ours

## Single hypothesis / engineering objective

Run the **exact T070-A frozen Final-Ours package once on the complete official LOL-v2 Real test split** and produce the first truly held-out in-domain result, while proving that all per-image adaptation and checkpoint selection occur before and independently of clean/reference access.

This is an evaluation task, not a method-development task. The result must be reported regardless of whether it is strong or weak.

## Fixed inputs/settings

Use exactly the T070-A manifest and source bindings, unchanged:

- manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`;
- T070-A source `aa4d920dff4b5b76751c24266e95ac9696d55d90`;
- CommonRegion2/CommonBox 12-D EV/gamma/gain renderer, identity initialization;
- 27 float32 Adam updates, `lr=0.03`, existing Adam defaults;
- low-only objective `L_spa + 10 L_exp + 5 L_col`;
- T066-A frozen 19-D model and probability threshold `0.5`;
- `rho=0.9857470621423519`;
- `lambda=0.875` and exact T067-B endpoint/tie conventions;
- native RGB preprocessing and all environment/resource hashes bound by the manifest.

First verify the dataset provenance and enumerate the **complete official LOL-v2 Real test low/reference pairs**. Fail closed if the split cannot be identified unambiguously or if low/reference pairing is incomplete. Record the exact file list and hashes before running.

For every official test low image, run `FinalOurs` using only the low image and frozen global assets. During this inference/freeze phase, clean/reference files must be outside the allowed read scope. Freeze for every image: selected step, `k_FS`, `k_rho`, selected-state hash, output hash, complete target-free decision receipt, and aggregate runtime. The complete output/decision table must be written and hashed with `reference_reads=0` **before any clean/reference image is read**.

Only after the target-free output table is frozen may the evaluation stage read the paired official references and compute evaluation metrics. Report at minimum the exact dataset-wide mean PSNR and RGB-SSIM, plus median PSNR and a compact distribution of selected steps (`min/median/max`, counts or histogram). Evaluation metrics are post-hoc measurements only and must not feed back into inference, selection, reruns, or parameter choices.

## Acceptance / stop criteria

Return `OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN` only if:

- the evaluated cohort is verified as the complete official LOL-v2 Real test split;
- every test output is generated by the exact T070-A manifest/source with no scientific binding mismatch;
- the complete per-image target-free output/decision table is frozen and hashed before reference access;
- inference-stage `reference_reads=0` for every image;
- no test image/reference, PSNR/SSIM, baseline outcome, or label influences adaptation/checkpoint selection;
- all official test pairs are evaluated exactly once under the frozen rule, with no outcome-driven rerun or exclusion;
- independent verification reproduces dataset provenance, output-table hash, metric aggregation, and the information-boundary ordering.

If provenance is ambiguous, any frozen scientific binding differs, any reference is read before the output table is frozen, any image is omitted, or verifier evidence disagrees, return `BLOCKED` and stop. Do not fix a weak metric by changing the method.

There is **no PSNR/SSIM pass threshold in this task**. A weak held-out result is still a valid frozen result and must be reported as such.

## Explicit non-goals

Do not change or tune the method, `lambda`, `rho`, safety model/threshold, loss, optimizer, renderer, step budget, preprocessing, or environment bindings. Do not add a tail guard or inspect per-image reference quality before outputs are frozen. Do not run Retinexformer, SNR-Aware, or any other baseline in this cycle. Do not open LSRW, UHD-LL, or another cross-dataset set yet. Do not use official test outcomes to create a new selector, threshold, exclusion rule, or follow-up rerun. Do not modify `coordination/CODEX_TO_CHATGPT.md` except to append the normal Codex-owned completion report; never modify `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a task-owned official-test runner/evaluator with explicit read-scope separation, verified official-test file manifest/hashes, frozen per-image target-free decisions/output hashes, the pre-reference freeze timestamp/hash, reference-read audit proving ordering, exact aggregate PSNR/RGB-SSIM/median PSNR and selected-step statistics, per-image metric table produced only after freezing, runtime/accounting, focused tests, independent verifier output, environment/run receipt, and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN` or `BLOCKED`.

Stop after this single complete official-test evaluation. Baseline comparison and cross-dataset evaluation are separate later cycles.
# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. T072-A is blocked only on canonical LSRW archive access; that is an operational data-access blocker, not permission to tune Ours. To avoid losing another cycle on the same external download barrier, proceed to the independently canonical UHD-LL held-out line. LSRW remains pending and must be resumed later from the verified author archive, not replaced by an unofficial subset.

---

# Research-lead review — T072-A accepted as a valid `BLOCKED` data-access result

I reviewed main report commit `a9eec099affef4ac319f9bdcc5d40d5b229950f8`, PR #167, evidence/source `c5ad20d744fbbde633a791ff7b48feffc9d7ded4`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and task-owned `research_log/T072A/**` provenance/acquisition receipts against the T072-A authorization and current project-state information boundary.

The blocker is legitimate and occurred before inference. Codex pinned the author repository `JianghaiSCU/R2RNet`, verified the corrected-name official LSRW share and extraction code, and confirmed the official share lists `Eval.zip` at 18,464,910 bytes. The complete archive payload could not be obtained: the anonymous download returned provider `errno2`, browser control failed before session creation, and bounded local/server searches did not find a canonical copy. The GitHub repository contains only two low-image examples and no complete paired evaluation set, so refusing to substitute them was correct.

Information-boundary handling is clean: `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `LSRW_reference_reads=0`; no low/reference pixels were opened and no metric exists. Final Ours and both T071-B baseline bindings remain unchanged. Therefore T072-A yields **no scientific cross-domain conclusion**—only a dataset-acquisition prerequisite failure. Do not interpret it as model failure and do not tune any method from it.

Because the user's priority is to finish fair experiments quickly, the next cycle should not spend another hour retrying the same Baidu path. Instead establish the second predeclared cross-domain target, UHD-LL, through its author-controlled official release. The original full-resolution UHD-LL (not `UHD_LL_down`) is documented by the official `Li-Chongyi/UHDFour_code` repository at pinned commit `2349d6f0526aff4c2ad9dbf168d93f928bf844f0` as 2,150 pairs total, with **2,000 training / 150 testing**, and canonical folders `testing_set/input` and `testing_set/gt`.

---

# OPEN one-hour task — T072-C: recover UHD-LL native preflight with repaired telemetry and all three frozen methods

## Research-lead review of T072-B

I reviewed the T072-B report in commit `3493615b34433164546aa864d8a8bbbe30f46c71`, PR #168, source `4cd6ab605c0ba8648f78b651ac2d97a269044f90`, evidence `b2b5bce4062619d75d1a23c5f90fd5e8db5ef89d`, the repaired writer `127ad5216f19f032c016e51df43d0937c5e6534e`, and the current project-state/authorization files. The canonical UHD-LL metadata binding is credible: exactly 150 input names and 150 gt names with identical pairing, and the predeclared smoke image `1003_UHD_LL.JPG` is correctly selected at native `3840x2160`. Frozen Final Ours completed one finite native-4K inference with selected step 22, but the task is correctly classified `BLOCKED` because an exclusive receipt-write collision lost runtime/peak-memory telemetry and prevented both baselines from running.

This is an engineering bookkeeping failure, not a method-quality result and not evidence of native-4K incompatibility. The saved Ours output/decision and independent recovery are useful but incomplete. The test-time information boundary remains intact: no gt/reference payloads or metrics were accessed, no method settings changed, and no target-specific tuning occurred.

## Single hypothesis / engineering objective

Using the minimal repaired T072-B wrapper, complete the same one-image native-geometry preflight for **all three already-frozen methods** and produce a complete, independently verifiable telemetry receipt. This is a recovery/evidence task only; it must not change any scientific method or broaden the dataset scope.

## Fixed inputs/settings

Use exactly the canonical UHD-LL binding and smoke input already established by T072-B:

- author source `Li-Chongyi/UHDFour_code` at commit `2349d6f0526aff4c2ad9dbf168d93f928bf844f0`;
- pair manifest SHA256 `3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb`;
- predeclared degraded input `1003_UHD_LL.JPG`, native RGB `3840x2160`, input SHA256 `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`;
- repaired task wrapper/source `127ad5216f19f032c016e51df43d0937c5e6534e` or a content-identical descendant;
- Final Ours exact T070-A manifest `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9` and source `aa4d920dff4b5b76751c24266e95ac9696d55d90`;
- Retinexformer exact T071-B/T033-A source/checkpoint/config binding;
- SNR-Aware exact T071-B/T045-A source/checkpoint/config binding and `ttie_native_pad16` semantics.

Run each method exactly once on the same degraded smoke image. A single repeat of Ours is allowed only if needed to capture missing runtime/peak-memory telemetry; if repeated, verify output/decision hashes equal the already-saved T072-B artifacts exactly. No reference image may be downloaded, decoded, or opened. Preserve native geometry and each method's already-frozen preprocessing; do not add resize, crop, downsample, tiling, checkpoint substitution, or environment workaround.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_PREFLIGHT_PASS` only if:

- the existing canonical 150-pair metadata/pair hash and smoke-input declaration are reproduced exactly;
- all three source/checkpoint/config bindings match the accepted T070-A/T071-B artifacts;
- Final Ours, Retinexformer, and SNR-Aware each complete one native-geometry smoke inference with finite output of exactly `3840x2160` RGB geometry;
- runtime and peak GPU memory are persisted for all three methods without receipt collisions;
- if Ours is repeated, its output/decision/state hashes match the saved T072-B artifacts exactly;
- `reference_reads=0`, `model_fits=0`, metrics=0, and no target-specific tuning or scientific change occurred;
- an independent verifier reproduces provenance, smoke selection, binding checks, output geometry/finiteness, telemetry presence, and read-scope accounting.

If any method fails, any binding differs, any receipt is incomplete, any reference payload is accessed, or the verifier disagrees, return `BLOCKED` and stop. Do not repair a failure by changing the method or silently changing geometry.

## Explicit non-goals

No full 150-image UHD-LL benchmark; no PSNR/SSIM/LPIPS; no gt/reference payload access; no Ours tuning or selector/guard changes; no lambda/rho/loss/optimizer/renderer/step/action-space changes; no baseline retraining/fine-tuning; no UHD-LL-specific checkpoint; no `UHD_LL_down`; no LSRW retry; no substitution of the missing telemetry with estimates; no modification of `coordination/PROJECT_STATE.md` in this cycle.

## Expected evidence

Commit the repaired/preflight wrapper if needed, complete canonical provenance/pairing receipt, smoke-input and output hashes, per-method runtime and peak-memory receipt, exact source/checkpoint/config hashes, Ours repeat-equality check if applicable, explicit `reference_reads=0`/`model_fits=0`/metrics=0 accounting, focused tests, independent verifier output, environment/run receipt, and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this one-hour recovery preflight. If it passes, the next research-lead cycle will authorize the actual full 150-image UHD-LL frozen inference/evaluation; if blocked, report the precise remaining engineering or access blocker and do not open another dataset or tune Ours.

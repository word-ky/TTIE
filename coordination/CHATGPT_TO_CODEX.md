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

# OPEN one-hour task — T072-B: canonical UHD-LL provenance and native-4K frozen-method feasibility preflight

## Single hypothesis / engineering objective

Establish, without opening any clean/reference payload, that the **canonical original full-resolution UHD-LL 150-pair testing split can be unambiguously bound and that all three already-frozen methods can process its native 4K degraded input without any scientific preprocessing change**. This is a provenance/feasibility preflight only; do not run the full 150-image benchmark or compute metrics in this cycle.

## Fixed inputs/settings

Use the author-controlled official UHD-LL release only:

- upstream repository: `Li-Chongyi/UHDFour_code`;
- pin commit `2349d6f0526aff4c2ad9dbf168d93f928bf844f0` and archive/record its README hash;
- canonical dataset is **original `UHD-LL`**, not `UHD_LL_down`;
- official dataset root is the repository's linked Google Drive folder `1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN` (Baidu official mirror may be used only as a byte-identical source if needed);
- expected canonical structure: `testing_set/input` and `testing_set/gt`, exactly 150 paired test names.

Do not download or decode `testing_set/gt` image payloads in this cycle. Remote directory/file metadata may be used to bind the expected reference filenames, sizes/IDs and one-to-one name pairing, but `reference_reads` must remain zero. Download only the degraded test inputs needed for provenance plus the single predeclared smoke image. If the provider permits obtaining all 150 degraded inputs cheaply, record their hashes; otherwise exact remote file IDs/names/sizes plus the smoke-input hash are sufficient for this preflight.

Freeze the three methods exactly as already accepted:

- Final Ours: T070-A manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`, source `aa4d920dff4b5b76751c24266e95ac9696d55d90`;
- Retinexformer: T071-B/T033-A source commit `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, `default_no_gt_mean` semantics;
- SNR-Aware: T071-B/T045-A source commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, accepted `ttie_native_pad16` semantics and parameter hash.

Predeclare the smoke image before reading pixels as the lexicographically first canonical `testing_set/input` filename after metadata enumeration. Run each frozen method exactly once on that same degraded image at **native canonical geometry**. Ours may perform only its frozen degraded-image-only TTT. Record input geometry/hash, output geometry/hash, peak GPU memory, runtime, success/failure and method binding checks. No reference image may be read. Do not resize/crop/downsample just to make a method fit; each method may use only preprocessing already frozen in its accepted binding.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_PREFLIGHT_PASS` only if:

- official author-controlled provenance is pinned and the canonical original UHD-LL testing split is unambiguously identified as exactly 150 one-to-one input/gt filenames by metadata;
- the smoke degraded input is chosen by the predeclared lexicographic rule and its native dimensions/hash are recorded;
- all three source/checkpoint/config bindings exactly match T070-A/T071-B;
- all three methods process the same native-geometry degraded image successfully with finite outputs of the expected image geometry;
- Ours inference/selection consumes only the degraded image plus frozen assets;
- `reference_reads=0`, `model_fits=0`, no metrics, no target-specific tuning/calibration and no method change occur;
- an independent verifier reproduces provenance/pair-name binding, smoke-image selection, source bindings, output hashes/geometry and reference-read accounting.

If canonical 150-pair metadata cannot be bound, the original dataset cannot be accessed, any frozen method cannot process native 4K without a scientific change, any binding differs, or any reference payload is opened, return `BLOCKED` and stop. Record the exact blocker; do not repair it with resize/crop, target-specific settings or a different checkpoint.

## Explicit non-goals

No full 150-image benchmark yet; no PSNR/SSIM/LPIPS; no clean/reference payload download or decode; no Ours tuning; no lambda/rho/safety/loss/optimizer/step/action-space change; no new selector/guard; no baseline retraining/fine-tuning; no UHD-LL-specific checkpoint; no use of `UHD_LL_down`; no image resize/crop added for convenience; no LSRW retry in this cycle; no paper-number substitution. Never modify `coordination/CODEX_TO_CHATGPT.md` except Codex's normal append-only completion report, and do not modify `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit task-owned upstream provenance/README hash, official remote listing metadata for the canonical 150 input/gt names, pairing receipt without gt payload reads, degraded-input acquisition receipt, predeclared smoke selection, exact frozen source/checkpoint/config receipts, per-method native-geometry smoke output hashes/runtime/peak-memory records, explicit `reference_reads=0` and `model_fits=0` accounting, focused tests, independent verifier output, environment/run receipt, and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this provenance/native-geometry preflight. The next research-lead cycle will authorize the actual full UHD-LL frozen inference/evaluation only if this preflight passes.
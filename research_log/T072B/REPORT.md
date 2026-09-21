# T072-B — BLOCKED

The canonical original UHD-LL 150-pair metadata and native4K smoke input are verified. Frozen Final Ours completed its one allowed native3840x2160 inference and saved a finite output/decision. However, an engineering receipt-write collision prevented final runtime and peak-GPU-memory telemetry from being persisted. The chained job then stopped before either baseline ran. Therefore the complete three-method preflight is **BLOCKED**, not UHDLL_NATIVE_PREFLIGHT_PASS.

This is a bookkeeping failure in the task wrapper, not demonstrated native4K method incompatibility, not a CUDA OOM, and not a scientific quality result. No references or metrics were accessed and no inference was repeated.

## Canonical provenance and smoke selection

Pinned author source: `Li-Chongyi/UHDFour_code` commit `2349d6f0526aff4c2ad9dbf168d93f928bf844f0`; README SHA256 `aad243530ac5db48fbd42c96b12ae9afacc3abfeed16641334a845c9e9b6cf48`. Its original UHD-LL link is https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN and describes 2,000 training /150 testing pairs. The downsampled UHD_LL_down release was not used.

Author-controlled folder traversal:

- dataset root `1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN`;
- testing_set `1CjTvAfXZlbR8V-wIeGquzE1JNULCkqmv`;
- input `1t8Zr6pBfcpCOtYzRAdxPGCjZcZq0VhGd`;
- gt `12kcm_mMhyHB19iVlfKg2gxTq3yLnIe6q`.

Drive HTML exposed only the first50 entries; complete public Drive file-list metadata returned exactly150 distinct names in each directory, with identical input/gt name sets and no continuation page. Saved `input_metadata.json`, `gt_metadata.json` and `pairs_manifest.json` contain names, remote file IDs, exact byte sizes and MIME types. Pair-manifest SHA256 **`3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb`**. Reading directory/file metadata did not download or decode any gt payload. No training data or second dataset was accessed.

The lexicographically first input, **1003_UHD_LL.JPG**, was declared at `2026-09-21T11:53:26.176705+00:00`, before acquisition at `2026-09-21T11:53:55.691204+00:00`. Remote ID `1icZifnPurmhZVzO02uNmODFMLyHrp-Ev`,752975bytes, SHA256 **`cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`**, JPEG RGB3840x2160. It was the only degraded image downloaded. `smoke_selection.json` and `input_receipt.json` preserve ordering. No resize/crop/downsample/tiling was introduced.

## Frozen methods and observed attempt

Original execution source **`4cd6ab605c0ba8648f78b651ac2d97a269044f90`**, authorization main **`ffb42f7f9727833250ea77804bd71eb3b318ce44`**, branch `codex/T072B-uhdll-native-preflight`. Task-owned wrapper reused the exact T070-A FinalOurs API and ReadScope plus unchanged T071-B Retinexformer/SNR exporters/bindings. Final-Ours manifest remains `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`, scientific source `aa4d920dff4b5b76751c24266e95ac9696d55d90`. All56 task-source bindings, original Ours manifest/code/assets and both accepted baseline asset sets were checked before the attempt. Original-source and baseline checks passed again before the final receipt write failed; saved-output recovery rechecked all execution-source bindings.

No optimizer/loss/renderer/selector/settings changed: 12-D CommonRegion2/CommonBox, 27 Adam updates at .03, low-only objective, rho .9857470621423519, lambda .875 and frozen safety model. Neither baseline checkpoint/config was changed; neither baseline inference ran.

| Method | Actual native smoke status | Precise inference time | Peak GPU memory |
|---|---|---|---|
| Final Ours | Output and decision saved;27 updates;selected step22;k_FS0;k_rho27 | Not persisted; unavailable | Not persisted; unavailable |
| Retinexformer | Not run after wrapper failure | Not measured | Not measured |
| SNR-Aware | Not run after wrapper failure | Not measured | Not measured |

Saved Ours tensor is float32 `[1,3,2160,3840]`, finite. Tensor-value hash `397ac007962709f6f24c210539fa9512b565bc6fb855b4a4abc2cc3634246af5`; output-file hash `b9c0438898c5d532125048404b05fc0837e6972094e085b7ddb7827950f7ea22`; decision-file hash `29a5df8505b2af5c6b9f64d1f856c1e708fe5778911e61ffcaebb8c423dd57a0`. The selected CPU output was cloned only for serialization to avoid retaining full-trajectory backing storage; no pixel/state computation changed.

A6000 GPU1 initial free memory was `3386179584` bytes after CUDA context initialization. That is **available memory, not peak use**, and no peak estimate is substituted. The GPU has48GB total; no other jobs were killed or changed.

## Failure, minimal repair and independent recovery

Run **`20260921-195909-ttie-t072b-uhdll-native`** started `2026-09-21T19:59:15+08:00` and exited1 at `19:59:47+08:00`. The task wrapper initially wrote `ours/receipt.json`, then attempted to use `research_log.T063A.common.write` for its final receipt. That helper uses exclusive mode `open('x')`, so the second call raised `FileExistsError`. Ours output and decision had already been saved. There is no model traceback or CUDA OOM; the run log preserves the exact filesystem error. The final in-memory seconds/peak statistics and final read ledger were lost when the process exited. The initial receipt's `attempts:0` is its pre-inference state, not a claim that the model never executed.

Minimal repair **`127ad5216f19f032c016e51df43d0937c5e6534e`** changes the initial filename to `initial_receipt.json`, leaving `receipt.json` for final exclusive creation. A direct check with the actual exclusive writer confirms both files coexist; changed modules compile. No retry or scientific method change was made. The original server release/failed-attempt binding is preserved. The branch binding now covers repaired wrapper code for a future separately authorized continuation; the original execution is reconstructed from the original source commit and archived source, not falsely attributed to repaired code.

Read-only `recover.py` checked exact150-pair metadata, lexicographic choice and acquisition ordering, the low hash, original execution binding, saved output shape/finiteness/value hash, and decision selection recomputed from saved target-free values/probabilities. It confirmed27 completed updates, step22, and absence of baseline output directories. **Saved-artifact audit PASS, task classification BLOCKED**: this audit does not recover the missing telemetry or claim all-method feasibility. Its missing seconds/peak fields are explicitly null. It never invokes a model or optimizer.

`reference_reads=0`, `reference_payload_downloads=0`, `model_fits=0`, metrics0, scientific reruns0. No LSRW retry or full150-image benchmark occurred. Input-only scope was enforced in the original Ours call; its final persisted read ledger is unavailable because of the receipt collision, which is disclosed rather than reconstructed as a measured ledger.

Validation: pre-run local10tests passed20.36s; remote10tests passed1.45s. Tests cover complete metadata pairing/selection timing and accepted Ours/baseline API/math/preprocessing contracts. Postrepair writer coexistence check PASS and syntax checks PASS. Independent saved-output recovery PASS with limited scope stated above. Nonfatal timm/OpenCLIP deprecation/default warnings preceded inference; no package upgrade or scientific workaround was applied.

## Execution and artifacts

Explicit release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t072b-uhdll-native`, outputs `/media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native`, source pinned in run command `TTIE_SOURCE_COMMIT`. Accepted environment preserved: Python3.12.12, torch2.4.0/CUDA12.1, numpy1.26.4, Pillow12.3.0, open_clip2.26.1, A6000; OMP/MKL/OPENBLAS threads1 and CUBLAS workspace4096:8. Exact command is in `evidence/run/run.sh`. Wrapper meta.releaseId remains a stale historical label; explicit cd and source pin are authoritative. Original metadata remains in raw archive; repo evidence only normalizes line endings.

- raw: `/media/wenchang/F/wjq/TTIE/shared/t072b/T072B_raw.tar`, 99573760bytes, SHA256 `0843ca901be0f14feea6e2b8c7d248d8c1ba3bb638523385b3496a8ec2c9ca0d`.
- recovery: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t072b/T072B_recovery.tar.gz`, 93464bytes, SHA256 `5822d836bfbc987f6104691ba5302083cd7ee2ce53a5203b3d54b394bb1fc82c`.

Full archive preserves the native output. Compact source/evidence/log recovery is present on both server roots and verified locally; task-owned metadata, decision, initial receipt, run logs, recovery audit and report are committed. Recovery-critical state/report/HANDOFF are under project-root research_log.

Recommended next step: research lead should issue a bounded recovery continuation that states whether to accept the existing Ours output with unavailable timing/memory or explicitly authorize a repeat solely to measure missing telemetry, and separately authorize the two still-unrun baseline smokes. Do not infer baseline feasibility from their lower-resolution tests. No full UHD-LL metric benchmark or method tuning is justified by this partial preflight. This stacked evidence PR is for task-file review, not a full-history merge recommendation.

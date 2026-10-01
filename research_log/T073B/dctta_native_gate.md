# T073B prospective PromptIR+DCTTA native-4K execution gate

2026-09-25, written before any UHD-LL target-image DCTTA model execution. Status: **design + synthetic execution evidence; target smoke not run**. Target model runs 0, reference reads 0, metrics 0.

## Official semantics (pinned clone `0526bf7b`, remote `/root/autodl-tmp/TTIE/T073B/source`)

- **Reported network = adapted student.** `train_test_promptir.py:270` sets `tta_model.model.eval()` and `:281-282` restores every test image with `tta_model.model` under `no_grad`. `tta_model.model` is the same object as the PromptIR passed to `tta.SRTTA` (`:188-192`, `tta.py:162`). The EMA teacher (`tta.py:196`, decay 0.45 at `:180`, update `:210-215`) only makes pseudo-labels (`:277-298`). The patch output returned by `SRTTA.__call__` (`:218-228`) is discarded by the main loop (`train_test_promptir.py:252`).
- **Two-phase, domain-level, one pass each.** After `set_seed(23)` (`:196`, `:44-51`): one shuffled Fisher pass over the whole target loader (`:239-241`, `tta.py:399-474`); then one shuffled adaptation pass over the same loader, `opt.iterations=1` step per 320×320 patch (`:244-252`, `tta.py:202-205`); then inference on all test images in sorted order with the final student (`:260-291`). The model is never reset between images: `reset_parameters` (`tta.py:479-481`) is never called. After every step `fisher_restoration` (`tta.py:366-382`) writes the source value back into the top-60% Fisher elements of each trainable weight/bias, so only the remaining 40% can drift, and they drift cumulatively. Adam moments, the EMA teacher, and the RDDM re-degradation network (`RDDM/src/residual_denoising_diffusion_pytorch.py:1313-1363`, 5 steps per patch, weights kept across images) are also cumulative.
- **Hyperparameters.** `options.py`: batch 1 (`:9`), lr 2e-4 (`:11`), betas (0.9, 0.999) (`:12`), fisher_ratio 0.6 (`:13`), num_workers 16 (`:17`), iterations 1 (`:38`), compute_fisher 1 (`:44`), teacher_weight 5 (`:45`). The patch size 320 is hard-coded at `train_test_promptir.py:206`. Trainable set: parameters whose top-level name contains `prompt`, `encoder` or `decoder` (`tta.py:51-64`); the optimizer receives only their `weight`/`bias` tensors (`tta.py:94-111`).
- **RNG.** `PromptTrainDataset_Simple.__init__` reseeds python/numpy/torch/CUDA with 42 (`utils/dataset_utils.py:811-812, 828-834`) after `set_seed(23)`. PromptIR (`:221`) and RDDM (`:235`) are then constructed and consume the CPU torch RNG before the first loader iterator draws its sampler seed and worker base seed. Crops use python `random` (`dataset_utils.py:841-842`). With workers, torch reseeds python `random` in each worker with `base_seed + worker_id` (`torch/utils/data/_utils/worker.py:223-224`, torch 2.3.0), so **worker count changes the crops**. It is fixed at the official 16.
- **Inference preprocessing/output.** `PairedImageDataset` center-crops to a multiple of 16 (`dataset_utils.py:707`, `utils/image_utils.py:60-65`); 2160 and 3840 are multiples of 16, so this is the identity. The model output itself is not clamped. The official path clamps only downstream: metrics clip to [0,1] (`utils/val_utils.py:53-54`), and PNG saving uses `clip(x*255,0,255).astype(uint8)` (`utils/image_io.py:383`). Static saved the raw unclamped float32 tensor. DCTTA saves the same raw tensor, so any downstream clip must be applied identically to both rows.
- **Side effects.** RDDM writes `model.pt` (~580 MB, overwritten) and 7 PNGs per patch under the cwd-relative `./results/` (`residual_denoising_diffusion_pytorch.py:1287, 1359-1363, 1436, 1448`). `WaveletTransform` falls back to `./wavelet.mat` (`utils/loss_utils.py:150-153`). The working directory is therefore a task-owned scratch directory containing the verified `wavelet.mat`.

## `run_low_only.py` versus official (discrepancies)

1. **Construction order (real, fixed in the new runner).** `run_low_only.py` builds PromptIR (`load_promptir`) *before* the train set, so the model's CPU-RNG initialization draws happen before the seed-42 reseed rather than after it. The sampler permutation and worker base seeds therefore differ from the official stream. This is a sampling-stream difference, not a change to any computation, but it is not the official order. `run_dctta_native_batch.py` restores the official order: seed 23 → train set (reseed 42) → loader → PromptIR → `.cuda()` → configure/collect/Adam/SRTTA → RDDM.
2. It saves 8-bit PNGs through official `save_image_tensor` (clamped, truncated) and has no 4K schedule. It is superseded for the table row by raw float32 outputs.
3. `torch.cuda.manual_seed(23)` is missing; on one GPU this is the same as `manual_seed_all`. The new runner calls both, as the official code does.
4. Differences that affect only memory or logging: the unused `origin_model` deepcopy (`:229`) is omitted, and the log is written to `<out>/adaptation.log`, not `./train.log`.
5. Identical: unchanged `tta.SRTTA`/`configure_model`/`collect_params`/`compute_fisher`, RDDM wrapper, all hyperparameters, `drop_last`/`pin_memory`/`shuffle`, workers 16, the logger passed positionally into SRTTA's `fisher` slot (as in the official code), and inference with the adapted student.

## Design of `run_dctta_native_batch.py`

- **Preflight (fail closed):** the receipt has 150 entries; the sorted low directory equals the receipt name order; all 150 low SHA256 match the frozen receipt; cwd `wavelet.mat` SHA256 is `8b153d20…`; free space is at least 96 MiB per output plus 1 GiB on the output filesystem and at least 2 GiB in cwd; the output directory must not already exist. No reference argument exists.
- **Adaptation:** the official order above, with the task-owned low-only loaders. `RecordedLoader` yields the unchanged batches and records `[basename, patch SHA256]` for the Fisher and adaptation passes. With `--expected-order`, any mismatch is an immediate assertion failure (`adaptation order drift`).
- **Handoff and cleanup (memory lifetime only):** adapted `state_dict` copied to CPU, asserted finite, hashed (`adapted_state_sha256`), and saved to `adapted_state.pt`. The parameter-update summary against the source checkpoint must show changes only inside the optimizer set and a nonzero changed count. Then the loader, SRTTA (teacher, Fisher masks, optimizer, source copy), RDDM and the training model are deleted, `gc.collect()` and `torch.cuda.empty_cache()` run, and allocated memory must be at most 256 MiB. A **fresh** PromptIR is loaded with that exact state (`strict=True`), and its state hash must equal the adapted hash. Only then is the sealed schedule applied: `schedule_depthwise(rows=256, threshold=250_000_000)`, `schedule_feedforward(rows=64, threshold=250_000_000)`, `schedule_skip_spill(threshold=1_000_000)`. The GPU layout at inference therefore matches the static row's fresh load.
- **Inference/output:** a per-image loop identical to `run_static_native_batch.py`, with the same row schema and per-image `output.pt.gz` + `decision.json`, asserting `[1,3,2160,3840]`, float32 and finite. At the end the state must still equal the adapted state. The manifest adds `promotable`, `adaptation_scope`, options/seeds, both orders, `expected_order_sha256`, adaptation seconds, adaptation peak reserved, post-cleanup allocated/reserved, inference seconds, inference peak reserved, source/adapted state hashes, `adapted_state_file_sha256`, `parameter_updates` (trainable tensors/elements/bytes; changed tensors/elements/bytes/names), `reference_reads: 0` and `metrics: 0`.
- **Sealed order procedure:** (1) `--order-only` on the canonical low set constructs exactly the same objects (so the RNG state is the same) but runs no model forward, and writes `adaptation_order.json`; (2) publish its SHA256; (3) the full run takes `--expected-order` and fails closed at the first drift. This works because nothing between construction and each loader iterator consumes the CPU torch RNG. The synthetic evidence below confirms it.

## Numerical caveats (inherited, not new)

Adaptation runs the unscheduled official forward on 320 patches (the schedule is applied only to the fresh inference model). Inference inherits the static schedule's operator equivalence, which is **not bitwise identical** to an unscheduled forward (synthetic max abs difference up to 2.1e-4, nonzero 8-bit mismatches; `native_schedule_gate_20260925.md`). The same schedule is used for both rows. GPU backward kernels are not forced deterministic, as in the official code, so the adapted state may not be bitwise reproducible across reruns; the synthetic repeat below measures this.

## Smoke definition (prospective)

`--smoke-one` is an **execution-only, non-promotable** smoke. It runs the Fisher pass and one adaptation step on one random 320 crop of canonical image 0 (`1003_UHD_LL.JPG`, `Subset([0])`), then the full cleanup, the fresh-model handoff and the native-4K inference of image 0. The manifest carries `promotable: false`, `adaptation_scope: smoke_image0_only_nonfinal` and the method suffix `-SMOKE-NONFINAL`. The output is never compared, inspected for quality or promoted; the domain-level adapted state comes only from the full run. The smoke exercises every target-specific failure mode (4K JPEG crops in 16 workers, RDDM/VGG/Fisher on target crops, the post-training CUDA context plus 4K inference memory) in about 1-2 minutes. Adapting all 150 images inside the smoke would spend a full adaptation to produce a state that the full run recomputes anyway. The full run also saves `adapted_state.pt` before inference and fails closed on image 1.

## Output location and stop conditions

- Smoke output `/root/autodl-tmp/TTIE/T073B/runs/dctta_smoke_one`, cwd `/root/autodl-tmp/TTIE/T073B/runs/dctta_smoke_work` (data disk).
- Full output `/root/TTIE_T073B_dctta_full` (root overlay; about 13 GB needed, preflight requires 15.1 GiB free), cwd `/root/autodl-tmp/TTIE/T073B/runs/dctta_full_work` (RDDM side files about 1.4 GB). Like the static outputs, it is instance-local and must be fetched and verified before teardown.
- Stop on OOM, any assertion (order drift, non-finite state/output, geometry/dtype, cleanup margin, state drift, low hash, disk), or a nonzero exit. There is no automatic change to precision, resize, tiling, schedule, workers or seed. A cleanup-margin failure or post-adaptation OOM returns to design review. Because the adapted state is saved, a separately sealed inference-only rerun is possible without new adaptation.

## Synthetic native-size GPU evidence

Completed 2026-09-25 18:36–18:45 +08 after the Ours-TTT GPU job exited, on 3 identical synthetic 4K PNGs (SHA256 `434f4814…`, names syn0-2) in `/root/TTIE_T073B_dctta_synth/`. All four jobs exited 0.

- `order16` (16 workers) order SHA256 `f0abe21fe4837648a7d1ebd9de12a68c2d5f1b594ac4aeb6692096d251b3a4cf`; `order0` (0 workers) `555cf00f3f022b119faf8902370ce74d8b842488a9b71b513b0afe1a18efb6d4`. Worker count changes the crop/order stream, so the official 16 is fixed. Fisher and adaptation passes use different shuffles (syn1,syn0,syn2 vs syn0,syn1,syn2).
- `full_a` and `full_b` both ran with `--expected-order` from `order16`; zero order-drift assertions, so the order-only preview equals the real run's patch sequence.
- Adaptation: 25.3 / 26.2 s for 3 images, peak reserved 23,381,147,648 bytes in both. Post-cleanup allocated 17,563,648 bytes, reserved 79.7 MB (limit 256 MiB).
- Scheduled 4K inference: peak reserved 49,165,631,488 bytes per image (+16.8 MB over the static row's 49,148,854,272), about 14.0 s per image; 94 depthwise and 47 FFN modules bound (same as static); every output finite float32 `[1,3,2160,3840]`; state unchanged after inference.
- Parameter updates (`full_a`): trainable 329 tensors / 8,596,503 elements / 34,386,012 bytes; changed 329 tensors / 3,435,857 elements (≈40%, consistent with 60% Fisher restoration) / 13,743,428 bytes; nothing changed outside the optimizer set.
- **Reproducibility (measured):** adapted state is **not** bitwise reproducible across identical reruns: `full_a` `6ec7439cfa44ba7e…`, `full_b` `8fa8bc687b6920ca…`; 328/548 tensors differ, max abs state difference 5.92e-4. Output difference for the same image: max abs 2.14e-4, mean 1.64e-5, 8-bit mismatch fraction 0.0041. This is the same order as the static schedule's operator-level difference. Sources: CUDA backward atomics (e.g. bilinear-interpolation backward in PromptGenBlock), `torch.topk` tie-breaking among equal Fisher values (`tta.py:457`), and RDDM noise/training; the official code does not force determinism either. Consequence: the full DCTTA row is a single-shot result that cannot be bitwise regenerated and must never be rerun after target outcomes.

Large synthetic outputs and adapted-state files were deleted afterwards to restore root-overlay space; `ab_diff.json` and manifests remain.

## Independent peer check (2026-09-25 ~18:50 +08)

A separate reviewer re-read the pinned official code and this runner and **confirmed** claims 1–7 (student-model inference, two-phase domain-level order and carry-over, construction order vs RNG reseeds, hyperparameters, fresh-model handoff computationally identical to `tta_model.model.eval()` because PromptIR has no dropout/BatchNorm/buffers/hooks/train-mode branches and `configure_model` only toggles `requires_grad`/`train()`, raw float32 output schema identical to static, no GT path including RDDM/wavelet side files, `--expected-order` consumes no RNG). Nothing blocked the smoke. Applied before any target execution:

- N1: a non-smoke, non-order-only run now **requires** `--expected-order`, so an unsealed full run cannot be marked promotable (new test).
- N2: the checkpoint SHA256 is asserted equal to the static row's `206baf0d…4d4a`.
- N6: before inference the runner asserts `torch.backends.cuda.matmul.allow_tf32 == False`, records cuDNN TF32/deterministic/benchmark flags and any `ACCELERATE_*` environment variables.

Disclosed assumptions (no code change):

- N3: low-only crop fidelity to the official paired loader assumes each GT image has the same size as its low image (the mirrored random draws use low dimensions). This holds for UHD-LL 4K pairs by dataset construction but cannot be checked without opening GT; it is an assumption, not a verified fact.
- N4: the official script's default checkpoint is the three-task `./pretrain/model.ckpt`; the five-task `epoch=80.ckpt` line is commented out (`train_test_promptir.py:215-217`). Using five-task is the prospective low-light-relevant choice already sealed for both PromptIR rows.
- N5: the source clone is not pristine: the recorded lazy-import patch moves the mmcv import inside `DCN_layer.forward` in `net/model.py`; PromptIR never uses DCN. The effective `PromptIR` class is the second definition at `net/model.py:1714`.
- Inference runs in the post-adaptation process (allocator history, accelerate/RDDM imported). With the TF32 guard this matches the static process's numeric settings; residual kernel-selection differences under memory pressure are possible and are covered by the non-bitwise disclosure above.

## Code SHA256 (LF)

- `run_dctta_native_batch.py` `48f9e5f141481b8e64b44ab3aacc76119de9689dd8ff07fb5363dcbce157c36c` (synthetic probe ran the pre-review version `a369b9c4…`; the post-review edits only add the N1/N2/N6 assertions and recorded fields)
- `test_run_dctta_native_batch.py` `098a1fa59a4e785261f6afce2c7576539aaba4b201d019a88c733c43e5610003`
- `probe_dctta_native_synthetic.py` `063229f59999df1b3941e98b7eb3567989463c98c68dac1869210b16f77369da`
- Reused unchanged: `run_low_only.py` `4ee1eb2c…`, `low_only_loader.py` `f7f8bd41…`, `chunk_conv_schedule.py` `1bd174c5…`, `spill_forward_schedule.py` `985c2b55…`.

Focused CPU suite (remote, `CUDA_VISIBLE_DEVICES=''`, `DCTTA_SOURCE_ROOT` set): 8/8 pass after the review edits (4 new, 4 existing; the GT-open mutation now runs in `--smoke-one` mode because unsealed full runs are refused first).

Status: **synthetic gate passed; one execution-only target smoke authorized** after this document is published. The full run additionally requires a published `--order-only` SHA.

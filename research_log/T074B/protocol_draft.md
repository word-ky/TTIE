# T074-B near-black target protocol draft (not sealed)

2026-09-25. Draft for the research lead and user. Nothing here is sealed until the preregistration commit. No target data exists on disk yet. Counters: target `reference_reads=0`, `metrics=0`, target model runs 0. Items marked **[DECISION]** need a user or research-lead decision before the step that depends on them. Items marked **[CHECK]** are facts to confirm when the data arrives.

Sources read: SNR-Aware `JIA-Lab-research/SNR-Aware-Low-Light-Enhance@1113144c` (`data/dataset_SID.py`, `dataset_SMID_test.py`, `dataset_SDSD_test.py`, `data/util.py`, `options/options.py`, `options/test/*.yml`, `test.py`); Retinexformer `caiyuanhao1998/Retinexformer@1e9a0efc` (`basicsr/data/{SID,SMID,SDSD}_image_dataset.py`, `Options/RetinexFormer_{SID,SMID,SDSD_*}.yml`, `Enhancement/test_from_dataset.py`, README); R2RNet `JianghaiSCU/R2RNet@cf59012d` README and arXiv 2106.14501 §III.

## 0. Phase order (user decision `research_log/T074A/ours_target_tuning_decision.md`)

1. **Stage and darkness check.** `--list-only` → fix list rules (§1–§3) → `stage_target.py` → sealed darkness statistic → pHash overlap (§6) → select targets → seal preregistration.
2. **Run, freeze and verify every non-tuned row** with `reference_reads=0`: RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, MR. Illuminate, QuadPrior, frozen-T070-A Ours-Step0, frozen-T070-A Ours-TTT (Step0 first; TTT abstention reuses this target's frozen Step0). Order per method: smoke → full → freeze/hash → `verify_generic_outputs.py`.
3. **Reference gate.** Only after an output-gate receipt lists every row of phase 2: open target GT, compute the preregistered metrics and statistics for the frozen rows. `stage_target.REFERENCE_DECODE_FORBIDDEN` stays `True` until this point. Phase-3 metric code is a separate module and must refuse to run without the gate receipt (not yet written).
4. **Ours target-tuning phase** (§8). Separate row "Ours-TTT (target-tuned)". Frozen rows stay as they are.
5. **Optional scene-disjoint two-fold check** (§8.4).

Baseline and frozen-Ours rows are never rerun after GT access.

## 1. Candidates: layout and test-list rule

The canonical loaders are the official SNR-Aware / Retinexformer test loaders. `stage_target.py` implements each rule and fails closed where the official code would silently mis-pair (it pairs folders by position after `sorted(glob(root/*))`; the stager requires the positional partner to have the same basename).

| Candidate | Layout (low / GT) | Test items | GT per item | Cluster |
| --- | --- | --- | --- | --- |
| SID-sRGB (Sony, SNR prep) | `short_sid2/<scene>/*.npy` / `long_sid2/<scene>/*.npy` | folders whose name starts with `1` (`dataset_SID.py:37-42`); **all** short files in each folder, every exposure (0.1 s, 0.04 s, …) | `sorted(long folder)[0]` (`dataset_SID.py:79`); shared by all shorts of the scene | scene folder |
| SMID (SNR prep) | `SMID_LQ_np/<seq>/*.npy` / `SMID_Long_np/<seq>/*.npy` | sequences in the test list; frames see §4 | first sorted GT file not containing `.ARW` or `half` (`dataset_SMID_test.py:47-52,83`); one per sequence | sequence |
| SDSD-in / SDSD-out (static, SNR prep) | `{indoor,outdoor}_static_np/input/<video>/*.npy` / `.../GT/<video>/*.npy` | videos in `testing_dir` of `options/test/SDSD_{indoor,outdoor}.yml` (in: 6, out: 10), plus folders whose `name.split('_2')[0]` is listed (`dataset_SDSD_test.py:43`); frames see §4 | per-frame GT, paired by index (`[idx:idx+1]`); stager also requires equal file names | video (base name before `_2`) |
| LSRW Eval | `<Eval>/{Huawei,Nikon}/{low,high}/*` | all 50 Eval pairs (paper: 5600 train / 50 eval; task brief: Huawei 20 + Nikon 30) | same file name in `high/` | image (no scene ids known) |

Test-list files: SMID uses `test_list.txt` (code reads `dirname(GT_root)/test_list.txt`, Retinexformer README calls it `text_list.txt` in `SMID_Long_np/`). The Retinexformer-linked Google Drive copy is saved as `smid_test_list.retinexformer_gdrive.txt`: 49 sequences, SHA256 `cd408769f80576bba9d4057add2d7c99e70bbf62682af4d7207885ab0d631fc3`. **[CHECK]** the copy inside the Baidu archive has the same hash.

**[CHECK]** at staging, from `--list-only` (paths only, no decode): item and cluster counts per candidate; SID long folders with more than one file (recorded as `gt_folder_file_count`; the official rule takes `[0]`); npy dtype/shape (stager fails closed on non-3-channel arrays). SID-Sony test counts from memory (~600 shorts, ~90 scenes) are not verified.

**LSRW darkness note.** The R2RNet paper says indoor low images were exposed longer "to avoid capturing extremely dark images". LSRW may well fail the sealed criterion; it is measured anyway and reported.

## 2. Evaluation geometry — recommend 960×512 for SID/SMID/SDSD, native for LSRW

**Recommendation:** for the npy candidates, the canonical low and (at phase 3) GT are exactly what the official loader returns: `np.load` → `cv2.resize(img, (960, 512))` (default `INTER_LINEAR`, on the stored dtype) → `astype(float32)/255` → `[2,1,0]` channel swap (`data/util.py:103-121,149-171`). All methods get the same 960×512 low; outputs stay at 960×512; GT goes through the same loader after the gate. LSRW has no official resize and stays at native geometry.

Why:
- Both official benchmark protocols evaluate SID and SDSD at 960×512 (`options/test/SID.yml`, `SDSD_*.yml` `train_size: [960, 512]`; Retinexformer `Options/RetinexFormer_{SID,SDSD_*}.yml`). Retinexformer also uses 960×512 for SMID (`RetinexFormer_SMID.yml:64`). SNR-Aware's `SMID.yml` has no `train_size`, which `options.NoneDict` turns into `None`, so SNR evaluates SMID at the stored npy size. **[DECISION]** recommend 960×512 for SMID too (Retinexformer convention, same geometry across targets). If the SMID npy files are already 960×512 the two agree; `source_shape` in the receipt shows this.
- SID native is about 12 MP. `research_log/T073D/native_gate.md:56` flags MR. Illuminate OOM risk above 8.3 MP; SNR-Aware and PromptIR would need their 4K memory schedules; compute would be ~25× larger.
- At 960×512 every method runs its official, unscheduled forward: PromptIR/DCTTA need no crop (960 and 512 are multiples of 16), SNR needs no pad, Retinexformer needs no pad, QuadPrior's official 512-short-side / multiple-of-64 geometry is exactly 512×960 (so its map-back is the identity size), MR. Illuminate's multiple-of-8 rounding is a no-op.
- Cost: the resize changes the SID aspect ratio (3:2 → 1.875:1) and low-pass filters the noise. It applies identically to low, GT and all methods, and it is the published protocol. Disclose it.

Metric-plan consequence: T073-A's "no resize" rule concerns outputs. For these targets the reference conversion becomes the official loader path above (uint8 resized GT / 255 in float32, then float64). **[DECISION]** amend `metric_plan.json` for T074 targets before sealing.

**LSRW geometry.** Resolution is not stated in the paper or repo. **[CHECK]** H, W at staging. If either is not a multiple of 16, the official PromptIR/DCTTA test loader center-crops (`low_only_loader.py:66-72`). The generic PromptIR/DCTTA runners fail closed in that case. **[DECISION]** only if LSRW is selected and not /16: recommend evaluating every row on the same PromptIR center-crop box (all methods keep their official computation), or else reflect-pad PromptIR input and crop back (unofficial for PromptIR).

## 3. Channel order and value range

- Official loaders treat the stored npy as BGR uint8 in 0..255 and swap to RGB. The stager does the same and writes 8-bit RGB PNG (lossless; round trip asserted). Float npy sources are cached as float32 `.npy`, but every runner requires PNG and will fail closed. **[CHECK]** npy dtype is uint8.
- **Risk [DECISION].** The SNR authors built SID/SMID sRGB with rawpy, which returns RGB. If they saved that array without conversion, the official `[2,1,0]` swap feeds BGR to every model. SID/SMID-trained models do not care; our LOL-trained baselines and Ours' CLIP gate do. This is low-only and method-independent to check: before any method runs, a human views about 10 brightened canonical lows in both channel orders (and/or the `input/` PNGs in Retinexformer's released SID/SMID results) and records which looks natural. Recommend: follow the official order unless that check is unambiguous; record the check either way. SDSD (video frames) is likely BGR as stored.
- LSRW: PIL decode, mode must be RGB, EXIF orientation must be absent or 1 (fail closed otherwise, because cv2 and PIL disagree on EXIF rotation).
- Darkness statistic (sealed, `darkness_criterion.md`): per-image mean of `0.299R+0.587G+0.114B` on the canonical RGB low in [0,1], float64 accumulation; pass iff median ≤ 0.10 and ≥ 80% of images ≤ 0.15. Reported per candidate: count, median, p80, p90 (numpy linear), fraction ≤ 0.15, pass. The list rule and geometry must be fixed before measurement because the statistic is defined on the canonical loader output (the resize barely moves a mean, but the item set matters).

## 4. Frame subsampling and compute

**[DECISION] recommend the SNR-Aware rule: first 30 frames per sequence for SMID and SDSD** (`dataset_SMID_test.py:54`, `dataset_SDSD_test.py:49-50`). Retinexformer uses all frames. First-30 is an official, deterministic, outcome-free rule from the data preparers; frames of a static sequence are near-repeats of one scene, so extra frames add cost, not independent evidence. SMID: 49 test sequences × 30 = **1,470** images (if every sequence has ≥ 30 frames). SDSD: 180 (in) / 300 (out). SID: all shorts of the test scenes (no official subsampling).

Per-image cost at 960×512 on the RTX 4090 48 GB (measured = synthetic smoke in this task; estimated otherwise):

| Row | s/image | Basis |
| --- | --- | --- |
| RetinexFormer, SNR-Aware, PromptIR | 0.29 / 0.23 / 0.62 | measured, §9 |
| PromptIR+DCTTA | 8.1 adaptation + 0.62 inference | measured §9; UHD-LL gave 1228 s / 150 = 8.2 s (patch-based, size-independent) |
| Ours-Step0 / Ours-TTT | 0.07 / 0.78 | measured §9 |
| MR. Illuminate (25-step DDIM inversion + sampling, fp32 autocast) | ~10–15 (estimate) | 50 SD-1.5 UNet passes at 120×64 latent; not measured |
| QuadPrior (10-step, fp16) | ~2–3 (estimate) | not measured |

Roughly 22–29 s/image for all eight rows, dominated by MR. Illuminate and DCTTA: SMID first-30 (1,470) ≈ 9–12 h; SMID all frames (~5k per the brief) ≈ 31–40 h; SID (~600) ≈ 4–5 h; SDSD-in+out (480) ≈ 3–4 h. MR. Illuminate/QuadPrior numbers must be replaced by their own synthetic probes at 960×512 before sealing. One-frame-per-sequence for SMID (49 images) is cheap but gives only 49 items and discards the official protocol; not recommended.

## 5. Statistical unit — cluster bootstrap

Items within a SID scene (shared GT, several exposures), an SMID sequence or an SDSD video are strongly correlated. Image-level resampling would understate CI width.

**[DECISION] recommend** keeping every T073-A endpoint and the image as the unit of the point estimates (mean/median PSNR, mean RGB-SSIM, paired deltas, strict win fraction with denominator N images), and changing only the resampling unit:

- clusters `c = 0..G-1` in sorted cluster-id order, sizes `n_c`, per-cluster delta sums `S_c`;
- `rng = numpy.random.Generator(numpy.random.PCG64(20260922))`; `idx = rng.integers(0, G, size=(10000, G), dtype=int64)` (C order), one shared matrix per target for all comparisons;
- resample statistic `sum_{c in idx_b} S_c / sum_{c in idx_b} n_c` (image-weighted, consistent with the image-mean point estimate);
- 95% percentile CI, quantiles 0.025/0.975, linear interpolation.

When every image is its own cluster (LSRW), this reduces exactly to the T073-A image bootstrap. Report G next to N. With G = 6 (SDSD-in) or 10 (SDSD-out) the percentile CI is unreliable; that counts against SDSD under the sealed "independence of scenes" selection criterion. **[DECISION]** LSRW scene grouping: no scene ids are published; treat pairs as independent unless file names reveal scenes.

## 6. Overlap check vs LOL-v2 Real Train

All candidates are separate capture campaigns (SID Sony α7S II, SMID, SDSD video, LSRW Nikon D7500 / Huawei P40 Pro), so overlap with LOL-v2 Real is unlikely; the sealed selection rule still needs the check before selection, so it must be target-low-only.

`phash_overlap.py`: 64-bit DCT pHash after exposure normalization (luma divided by its 99.5th percentile, clipped). This makes a near-black low and a normal-light image of the same scene hash alike (unit test: a 5%-brightness, noisy, resized copy stays within the threshold; unrelated scenes stay above it). Compare every canonical target low against LOL-v2 Real **Train Low and Train Normal** (source data, may be decoded). Flag Hamming ≤ 10 for human review of the low images only. Target GT is not used: raw-byte hashes cannot match across npy/PNG formats, and GT pHash would need a decode. Within-target, the stager fails closed on identical GT bytes across clusters and on any low identical to a GT. Low-only is sufficient because a duplicated LOL pair would show its scene in the target low; after the gate a GT-side pHash may be reported as disclosure only, never for selection.

**[DECISION/INPUT]** LOL-v2 Real Train is not on the GPU host or this PC; give its location (or approve downloading it).

## 7. Method execution on the new targets

All generic runners take `--low-dir --low-receipt --expected-count --out`, derive H×W from the receipt, write lossless float32 `[1,3,H,W]` `output.pt.gz` + `decision.json` + `output_manifest.json` (same row schema as UHD-LL: `low_name, low_sha256, shape, dtype, output_tensor_sha256, output_file_sha256, whole_run_seconds, peak_gpu_memory_bytes`), with `reference_reads: 0`, `metrics: 0`. Runners never receive the reference-opaque manifest.

- **PromptIR static, PromptIR+DCTTA, Ours-Step0, Ours-TTT:** `make_generic.py` derives them from the frozen UHD-LL runners by a declared list of text substitutions; `test_verify_generic.py` re-derives them and requires byte equality (on the GPU host, against the originals that produced the UHD-LL rows). Only constants change: count 150 → `--expected-count`, 2160×3840 → receipt H×W, the pinned UHD-LL Step0 manifest hash → `--expected-step0-manifest-sha256`, plus a fail-closed /16 check for PromptIR. DCTTA keeps the official construction order, seeds, `--order-only` sealing of the domain-level adaptation order, and `--expected-order` for promotable runs. **[DECISION]** DCTTA domain = the whole target test cohort (for LSRW: Huawei + Nikon as one domain), as for UHD-LL.
- **Memory schedules:** PromptIR's 4K depthwise/FFN/spill schedules and SNR's 512-query-row attention are opt-in (`--memory-schedule`, `--query-row-schedule`). Default is the official unscheduled forward, which fits easily at ≤ 1 MP (smoke peaks in §9). Plan §6 allows validated schedules only "where geometry requires them"; at 960×512 it does not. Note that at 960×640 or larger the PromptIR depthwise threshold (2.5e8 elements) would have activated the UHD-LL schedule if left on; keeping it off avoids a silent code-path change.
- **RetinexFormer / SNR-Aware:** `run_retinex_snr_generic.py` imports the accepted T072-AZ exporters' `load_model`/`forward` unchanged after checking every bound file hash (bindings SHA256 `36cd7cdf…`, Retinexformer arch file `1567c89d…`, SNR parameter hash `11d3d667…` before and after), and decodes the PNG with the exporters' own lines.
- **Process environment** matches the frozen executions: Ours and PromptIR/DCTTA run with `CUBLAS_WORKSPACE_CONFIG`/`*_NUM_THREADS` unset (FinalOurs asserts its bound environment); RetinexFormer/SNR with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and single-thread BLAS as in T072-AZ.
- **Ours-TTT:** Step0 is run and verified on the target first; the no-active abstention reuses this target's frozen Step0 bytes, exactly the sealed UHD-LL rule.
- MR. Illuminate / QuadPrior: generic runners exist in `research_log/T073D`, `T073E` (commit 04c535c9), not touched here. Their receipt interface (`files[].name/sha256`, low dir holding exactly the receipt files, unique stems) matches `low_receipt.json`.

## 8. Phase 4: Ours target tuning (user decision; separate, disclosed row)

### 8.1 Tunable knobs in frozen T070-A (host source `/root/autodl-tmp/TTIE/T073C/recovery/source`)

| Knob | Frozen value | Where |
| --- | --- | --- |
| Gate activation threshold `q_joint` (region active iff evidence > q_joint; the no-active abstention follows from it) | 1.053775 (95th pct of clean calibration maxima) | `T022A_gate.json`; `ttie/joint_gate.py:30-32` |
| Gate calibration `tau` / `scale` per axis | tau [0.02742, 0.00167], scale [0.07507, 0.05785] | `T022A_gate.json` → `FixedObjective`, `ttie/semantic_ttt.py:91` |
| Action box: EV upper bound for dark active regions; gamma lower bound | 2.0; 0.5 | `ttie/ev_range_box.py:10`; `ttie/gamma_range_box.py:9` |
| Optimizer / budget | Adam lr 0.03, betas (0.9, 0.999); 27 updates / 28 states | `research_log/T062CR2/core.py`; `T070A/constants.json` |
| Loss weights (spatial, exposure, color); exposure target; pools | [1, 10, 5]; 0.60; 4 / 16 | `research_log/T062A/core.py:6-14` |
| Stop selection: rho; lambda; probability threshold | 0.985747; 0.875 (grid 0…1 step 0.125); 0.5 | `research_log/T067B/core.py` |
| Step-probability model | logistic, 19 features | `research_log/T066A/evidence/model.json` |
| Region grid | 2×2 (`CommonRegion2`) | `ttie/common_gain.py` |

### 8.2 Search and logging
**[DECISION]** fix the search space before phase 4 starts. Proposal: a small declared grid (e.g. `q_joint` × {frozen, lower percentiles 90/80 of the same calibration}, lambda over the existing 9-point grid, exposure target {0.5, 0.6, 0.7}, updates {27, 40}); anything else is a new decision. Every setting is logged to an append-only JSONL: setting, code/manifest hash, per-image outputs hash, mean/median PSNR, mean RGB-SSIM, abstention count, runtime.

### 8.3 Selection rule
Choose the setting with the highest image-mean PSNR on the target; ties (±0.01 dB) broken by mean RGB-SSIM, then by distance from the T070-A defaults. Report it as "Ours-TTT (target-tuned)" with the number of settings tried, next to the frozen Ours-Step0/Ours-TTT rows. Its CIs are optimistic (selection on the same images); say so.

### 8.4 Optional scene-disjoint two-fold
Split clusters by sorted cluster id into alternating folds (fixed before phase 4). Tune on fold 1, evaluate on fold 2, swap; report the pooled out-of-fold result next to the same-image tuned row.

## 9. Synthetic execution smoke (this task)

Synthetic inputs only (`make_synthetic_targets.py`): set A = SID-like npy (1424×2128 uint8) staged to 3 lows at 960×512; set B = LSRW-like PNG, 3 lows at 600×400. Driver `run_synthetic_smoke.sh`, results under `/root/autodl-tmp/TTIE/T074B/runs/synthetic_smoke/`. Steady-state s/image and peak reserved VRAM at 960×512, official unscheduled forwards:

| Row | s/image | peak GiB | verifier |
| --- | --- | --- | --- |
| RetinexFormer | 0.29 | 1.85 | pass (A, B) |
| SNR-Aware | 0.23 | 1.48 | pass (A, B) |
| PromptIR | 0.62 | 5.79 | pass (A); B fails closed at the /16 preflight, as designed |
| PromptIR+DCTTA | 8.1 adaptation + 0.62 | 5.44 | pass (A); B fails closed before adaptation |
| Ours-Step0 | 0.07 | 0.67 | pass (A, B) |
| Ours-TTT | 0.78 | 0.74 | pass (A, B); all 6 synthetic images gate-active, so the abstain path is covered by the CPU tests only |

The generic RetinexFormer/SNR outputs are bitwise identical (HWC SHA256) to the original T072-AZ exporters run on the same PNGs (`runs/exporter_equivalence/`).

## 10. Open items before sealing

1. Geometry for SMID (960×512 recommended) and the metric-plan amendment for official-loader GT conversion.
2. SMID/SDSD frame rule (first 30 recommended).
3. Cluster bootstrap as the resampling unit.
4. Channel-order check for SID/SMID.
5. LOL-v2 Real Train location for the pHash check.
6. DCTTA domain definition for multi-camera LSRW; PromptIR /16 handling if LSRW is selected and not /16.
7. How many passing candidates enter the main table (the darkness file fixes the order, not the number).
8. Phase-4 search space and selection rule; whether to run the two-fold check.
9. Phase-3 gate receipt and metric module (to be written; must check the gate receipt and use the reference-opaque manifest hashes to confirm GT bytes are unchanged before decoding).

## Orchestrator defaults adopted under the user's full delegation (2026-09-26 early, +08)

The user delegated execution fully while unavailable. Until the user or research lead overrides, and before any method runs on real target data: evaluation geometry 960×512 through the official SNR loader for SID/SMID/SDSD and native for LSRW; SMID/SDSD first 30 frames per sequence; cluster bootstrap (seed 20260922, 10,000 resamples); channel order decided by visual inspection of ~10 brightened target LOW images only (no GT, no method outputs), recorded before any run; pHash overlap against LOL-v2 Real Train if that set can be obtained, otherwise recorded as not performed.

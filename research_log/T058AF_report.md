# T058-AF — frozen source-reference alignment verdict

**DONE: frozen-energy detail tangent not ready.** Both preregistered global thresholds fail. This is an isolated source-training diagnostic; it does not authorize deployment or establish target-domain performance.

Authorization `e11276ca3df71be59b423719317cd7eb992dd76d`; immutable Stage-A accepted head `3110bcbfcb1c018504f05a1f02052f1db59d3b33`; tested source `40642d35a4ab78d686f41673b6b933c7a39a6827` with 208 exact Git-blob/source bindings. PR #95.

## Result

| Quantity | Result | Frozen gate |
|---|---:|---:|
| Canonical rows | 7,346 | All retained |
| Nondegenerate rows | 7,244 | Existing norm > 1e-12 convention |
| Positive dots (nondegenerate) | 4,959/7,244 = 0.684566537824406 | >= 0.75: FAIL |
| Median cosine | 0.230516276093197 | >= 0.50: FAIL |
| Mean cosine | 0.178650710307290 | Descriptive |
| Cosine p10 / p90 | -0.423131478618847 / 0.670138080035154 | Descriptive |
| Degenerate rows | 102 (1.3885107541519195%) | Retained, excluded only by unchanged summarizer |
| Learned-gradient zero rows | 98 | Existing <=1e-12 convention |
| Reference-gradient zero rows | 102 | Existing <=1e-12 convention |

The full 64-D T029 alignment and T058 aggregate/classify functions were reused unchanged. Positive-dot fraction and cosine statistics use nondegenerate rows exactly as the accepted summarizer does; zero rows remain in every artifact. No subgroup rescue, reweighting, subset choice or alternate gate.

## Execution and boundary

Sole run `20260917-153308-ttie-t058af-stageb`, release `20260917-153229-ttie-t058af-stageb`, physical GPU1 / NVIDIA RTX A6000, torch 2.4.0+cu121, CUDA 12.1. Started 2026-09-17T07:33:18.318885+00:00; completed 2026-09-17T07:35:55.459450+00:00; measured 157.140596209 s; launcher exit 0. Local 8 tests passed in 5.24 s; server 8 tests passed in 1.60 s.

Before the first clean source open, all 7,346 immutable learned gradients were reopened: both shard file/receipt/manifest hashes, authoritative AE completion marker, every tensor hash/norm/shape/dtype/finite flag, and exact canonical coverage verified at 2026-09-17T07:33:31.379596+00:00. First source open 2026-09-17T07:33:31.636838+00:00; last 2026-09-17T07:35:39.464555+00:00. Exactly 80 unique `train_t014_sobolev` files were opened, each with filename/split/manifest hash and timestamp; other image paths were denied. All Stage-A files and stored gradients verified unchanged again after reference computation.

Each row reconstructs the unchanged legacy state and exact T054 Detail renderer. Raw/y0 hashes match Stage A and original preflight; all detail and legacy tensor hashes match Stage A before and after differentiation. The source scalar is exactly `(output.double()-clean.double()).square().mean()` from original T058-A Stage B, with GPU float32 coordinate/renderer/reference gradient. Exactly one autograd call per state; no obsolete FD/JVP. 115 atomically persisted reference chunks were reopened to verify hashes and regenerate all per-row alignment values. Local compressed-CSV replay independently confirms canonical ordering, counts, positive-dot fraction and median.

Zero learned-gradient recomputation, optimizer updates, persistent scientific-state changes, FD, target-domain/LOL-v2 access, official-test access, PSNR/SSIM selection or deployable rollout. CLIP/energy/prototype checkpoint files and all source-bank/source bindings verified unchanged; scoring models were unnecessary for reference differentiation and were not loaded or updated.

## Descriptive condition results (not gates)

| Condition | Rows | Nondegenerate | Positive-dot fraction | Median cosine |
|---|---:|---:|---:|---:|
| clean | 218 | 140 | 0.6571428571428571 | 0.20825934062879475 |
| homogeneous_bright | 1644 | 1632 | 0.8278186274509803 | 0.4566476802757094 |
| homogeneous_dark | 1736 | 1728 | 0.5167824074074074 | 0.020976333852950644 |
| left_right | 1874 | 1872 | 0.7350427350427351 | 0.2629758095511302 |
| quadrants | 1874 | 1872 | 0.6661324786324786 | 0.1895563023109837 |

Full descriptive image/state stratifications are in `T058AF_summary.json`; no subgroup selects or changes the global verdict.

## Artifacts and deviations

`T058AF_rows.csv.gz` holds all 7,346 row identities, g_E/g_R hashes/norms, dot/cosine/degenerate/positive flags, raw/y0 hashes and verification. `T058AF_manifest_index.json` binds every reference chunk and full raw manifest; complete/receipt/source-opens/Stage-A-verification/reopen/summary/export receipts and exact run command/log accompany it. Full tensors and before/after state dictionaries are in `/home/wenchang/asdasdsad/wjq/TTIE/shared/t058af/T058AF_evidence.tar.gz` and `/media/wenchang/F/wjq/TTIE/shared/t058af/T058AF_evidence.tar.gz` (4015024 bytes; SHA256 `29d7735ad9b3e3ed4cc857f482aa11821d0bd9161db48c8b97f037a1314d9760`), verified identical. Due observed critically low D-disk space, full raw archive is retained on server home/F instead of duplicated locally/in Git; hash-bound compressed per-row CSV and receipts are committed. Scientific evidence is not reduced or filtered.

No scientific failure, rescue or rerun. Known NVML driver/library warning did not prevent CUDA execution; no driver change. Inherited PR integration/history remains unchanged. Stop for research-lead review: the current frozen energy does not meet the source detail-tangent readiness gate. Do not begin retraining or real-domain integration in this cycle.

Publication infrastructure note: local Git add failed with No space left on device after evidence was fetched. No experiment rerun. The same artifacts are published through the GitHub Git-data API; remote home/F project copies remain the recovery source.

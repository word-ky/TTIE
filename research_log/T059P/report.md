# T059-P — DONE

2026-09-18T09:12:12.292441+00:00

Classification: **frozen CLIP-latent locality fails the source control; stop**.

Authorization `89f0baabb74530724c9c78e6014f2d42e15cbcab`; source `3739020b25e6be511c0871813f710911d59be064`; branch `codex/T059P-clip-latent`. Sole run `20260918-170231-ttie-t059p-clip`, release `20260918-ttie-t059p-clip`; 09:02:37–09:10:50 UTC, exit 0. GPU work used physical GPU 1, NVIDIA RTX A6000.

| Partition | Images / banks / rows | Huber | Margin to 0.07650849781930447 |
|---|---|---|---|
| Fit leave-one-image-out | 40 / 200 / 3604 | 0.11241389811038971 | -0.03590540029108524 |
| Reused source selector | 8 / 40 / 753 | 0.19629433751106262 | -0.11978583969175816 |

The first applicable gate fails on fit. No claim that prompt compression is the established bottleneck; no training, alternative representation or follow-on evaluation was attempted. Wait for research-lead review.

Exact T059-N split and fit-only scalar normalization were retained. The accepted T014 `FrozenCLIP.from_checkpoint` and `image_embeddings` code bytes were reused with checkpoint `ViT-B-32 / laion2b_s34b_b79k`, SHA256 `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`, at `/home/wenchang/asdasdsad/wjq/TTIE/shared/t004/open_clip_pytorch_model.bin`. All 240 selected `bank_images.pt` hashes match the accepted training manifest; no raw source or separate clean/reference tensors were opened.

The existing API returns **5 × 512** normalized embeddings per state (full image, TL, TR, BL, BR). All five fixed views were retained in order and losslessly flattened, giving e dimension 2560 and z dimension 5120. No view selection/averaging/sweep. Exactly z=concat(e-e0,e0), unique same-bank state-0 anchors, squared Euclidean FP64 k=1 and smallest canonical global-ID exact tie break. Feature forwards used FP32, eval/no_grad, one state/five views per call: **4357 state-image calls / 21785 view embeddings**. The original fixed text-prototype initialization inside from_checkpoint was unchanged; no text variant was selected. Preprocessing/config/code hashes and exact input bindings are in provenance.json/result.json.

Staging UTC: maps frozen `2026-09-18T09:03:31.999127+00:00`; first fit scalar read `2026-09-18T09:03:34.728773+00:00`; both prediction tensors frozen `2026-09-18T09:03:34.937078+00:00`; selector scalar opened `2026-09-18T09:03:36.733671+00:00`. The source and input hashes before/after match. All requested protected-access counters are zero (including training/optimizer/reference-image forwards, early selector access, inner-held, outer, new cohort, reference gradients, target-domain, LOL-v2 and official test).

Validation: **4 focused tests passed in 1.43s**, including clean-reference replacement/absence invariance, full-view anchor algebra, image exclusion, canonical ties and first-applicable classification. Independent CPU NumPy replay verified every 4357 representation, neighbor, tie, distance, prediction, loss and per-image/per-bank summary from persisted embeddings (no repeated CLIP forwards); maximum distance error 1.7763568394002505e-15 on each partition. Fit tie queries 7 (maximum 3); selector ties 0 (maximum 1). Per-query donor IDs/distances/counts/ties are in maps.pt; per-image/bank Huber in result.json.

Commands: project venv `python -m pytest -q -p no:cacheprovider --import-mode=importlib research_log/T059P/test_core.py`; staged separate processes `python -m research_log.T059P.run maps`, `predict`, `evaluate`, then `python research_log/T059P/verify.py`, with the run artifacts path as --out/positional argument. Source contains core.py, run.py, tests, verifier, authorization and provenance/source bindings; no scientific source edits after the pinned commit.

Result SHA256 `413fa8a6ea3f2ea34b12385db670e63c1777fff203f08352befb83d536acae3e`; vectors `0f53f597a9c319747678f498f3db99cdd3bd69f117cff26d96d1fa73cb7298b0`; maps `19e705497bec2f01fd11ee3dc889c097b74af76ffdb44fafa56e99bb1744eed1`; predictions `8e2414c88be15f1c59199acb8131bf2fc68c0abcaa5462edc7e9793d25121230`. Individual tensor hashes are recorded in the freeze markers. The 178,676,960-byte vectors.pt is delivered losslessly as gzip parts; ARTIFACTS.md reconstructs and verifies it. Full raw evidence remains on both remote project disks.

Observed infrastructure issues: local D: is full, so no local artifact writes; source/evidence were published through the authenticated local GitHub API without placing credentials on the server. One authorization-copy attempt exceeded Windows command length, then compressed transfer encountered one SSH timeout; the retry succeeded before launch. Bare remote python is unavailable; project venv works. Existing NVML mismatch and upstream deprecation/load warnings did not prevent PyTorch CUDA operation. No scientific rerun, changed gate or fallback model.

**Prior review correction:** T059-O's pinned actual fit/selector Huber values are **0.04802930727601051 / 0.1556149274110794**, not the two values transcribed in the lead review. The selector is reused from N, not fresh. Original O artifacts and lead-owned files remain unchanged. This correction does not change O's classification.

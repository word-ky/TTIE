# T059-L — DONE

bank-relative feature displacement is not even source-LOO consistent under fixed 1-NN; stop.

Source `7b9f68342735d8572452e53b4a0fd5efc6811f33`; authorization `2a919598f6d60e588c646ba0afcedf13c0841697`.

| Metric | Train-LOO | Inner-held |
|---|---:|---:|
| huber | 0.09940164536237717 | 0.23254923522472382 |
| margin | -0.0228931475430727 | -0.15604073740541935 |
| absolute_baseline | 0.06445551663637161 | 0.23172274231910706 |
| change_from_absolute | 0.034946128726005554 | 0.0008264929056167603 |
| queries_with_ties | 482 | 145 |
| maximum_ties | 235 | 240 |
| candidate_count_min | 4237 | 4357 |
| candidate_count_max | 4306 | 4357 |

Fixed gate 0.07650849781930447. Train-LOO fails the first criterion; held value is descriptive only. This coordinate change is not supported under the prescribed fixed 1-NN. No follow-up recipe attempted.

Exactly 240 train and 80 held banks each have one state_index==0 anchor. Anchor and dx tensors are saved in anchors_dx.pt with per-tensor hashes. dx subtracts the anchor in the existing float32 standardized coordinate; direct float64 squared Euclidean distances reuse T059-G nearest(), with canonical global-ID ties. Train excludes its own image; held uses all 4,357 fixed training candidates. No K oracle donor identity was used.

Anchors persisted 2026-09-18T03:32:52.595582+00:00; complete maps persisted 2026-09-18T03:32:53.228407+00:00; scalar evaluation opened 2026-09-18T03:32:54.915457+00:00. Anchor/dx SHA256 517699e3ed0c1668ce179c734c41e260d0ef74706ab567a20c38b868caf18a14; maps SHA256 5e92ba354b70e5bbc66d76bab663bc63de997780f114f6cfc1922e9c99a4ddfe.

Accepted G/K result and source bindings, G feature rows and E split metadata replay exactly. Initial baseline check uses SHA-bound recorded aggregates; exact rowwise G Huber replay and mixed-scalar full-file hash occur only after map freeze. Scalar reader accesses selected target/prediction storage, never gradient tensors. Before/after input and source hashes unchanged. Outer rows excluded; all prohibited-access/training counters zero.

Tests: 4 passed in 1.45s. Sole run 20260918-113246-ttie-t059l-relative on physical A6000 GPU1, 03:32:51–03:33:00 UTC, exit0. Independent CPU NumPy recomputes all 5,886 anchors, dx vectors, neighbor IDs, ties, candidate counts, predictions and losses: PASS. Maximum distance differences 8.881784197001252e-16 / 1.7763568394002505e-15; exact neighbor/tie results. No scientific failures or reruns. Nonblocking NVML warning. Local D ENOSPC prevents fetch/full artifact copies; project records preserved remotely and published through GitHub API.

Commands: `python -m pytest research_log/T059L/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059L.run maps --out "$AUTODL_RUN_DIR/artifacts/T059L"`; `python -m research_log.T059L.run evaluate --out "$AUTODL_RUN_DIR/artifacts/T059L"`; `python research_log/T059L/verify.py "$AUTODL_RUN_DIR/artifacts/T059L"`. CUDA_VISIBLE_DEVICES=1; OMP/MKL/OPENBLAS threads=1.

Stop for research-lead review. No representation redesign or deployment claim.

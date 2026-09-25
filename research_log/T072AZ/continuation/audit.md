# T072-AZ continuation — frozen SNR-Aware memory-path audit

The exact T027B official source is the one hashed in `../seal/baseline_bindings.json` and independently rechecked by `../verify_port.py` before T072-AZ smoke. This audit reads only those source files and prior smoke receipts. UHD-LL reference/clean payloads remain unopened.

## A — official/repository-supported high-resolution protocol

Not established for native full-resolution evaluation. Official `README.md:68-111` routes LOL-v2 Real to `test_LOLv1_v2_real.py`; that script calls `model.test4()` at line 76. `models/Video_base_model4_m.py:213-220` explicitly resizes the input and mask to 400×608, infers there, then interpolates output back. `test_LOLv2_synthetic.py:76` instead calls `test5()`, which resizes to 384×384 at `models/Video_base_model4_m.py:260-267`. `test.py:76` calls `test()` with no resize but provides no high-resolution memory schedule. `data/dataset_LOLv2_real.py:117-121` reads native LQ and GT for the test phase; the latter is prohibited for this low-only task. A scoped search over frozen `.py`/`.md` files found no official chop, tile, overlap, or high-resolution inference utility. The official LOL-v2 path is therefore not a same-native-output solution.

## B — execution-equivalent query-row scheduling

Established prospectively by source structure and synthetic equivalence tests. `models/archs/low_light_transformer.py:55-76` creates full-image tokens and a key mask. At UHD-LL 2160×3840, the two stride-2 convolutions and 4×4 unfold yield 135×240 = 32,400 tokens. `models/archs/transformer/SubLayers.py:31-55` projects q/k/v and broadcasts the same mask across all query rows. `models/archs/transformer/Modules.py:15-22` computes scaled qkᵀ, the same masked fill of -1e9, softmax over all keys, and multiplication by v. Each output query row depends on all keys/values but not on any other output query row. `models/archs/transformer/Models.py:67-71` discards the returned attention matrix; its six encoder layers consume only the output feature tensor.

Task-owned `chunk_attention.py` computes exactly those same operations in fixed blocks of 512 *query* rows while retaining all 32,400 keys and values. It neither tiles the image nor changes context, model parameters, checkpoint, input, mask, normalization, precision, source training, or adaptation. Only the unused attention-matrix return is omitted. The original dense attention matrix would occupy 8×32,400²×4 = 33,592,320,000 bytes (31.29 GiB), agreeing with the observed 31.29 GiB allocation failure. One 512-row attention block needs 530,841,600 bytes (0.494 GiB) before transient copies. This is a memory-scheduling change, not a scientific/protocol change.

`test_chunk_attention.py` compared the frozen official attention and the chunked implementation on nine deterministic float32 synthetic cases (31/513/1025 tokens × no/mixed/all-zero masks), including chunk boundaries. Every case passed `atol=1e-6, rtol=1e-5`; maximum absolute difference was 1.1921e-7. On the paid RTX 4090, `test_model_equivalence.py` loaded the accepted checkpoint and compared full-model outputs on a synthetic 64×64 float32 input before/after the patch; they were identical (`max_abs_delta=0.0`, `synthetic_model_receipt.json`). No UHD-LL image or reference was used. One authorized UHD-LL smoke is the next validation step.

## C — prohibited alternatives

The official resize-and-upsample test4/test5 paths change the native output; evaluation crops, patch/tile prediction with reduced context, FP16/AMP, checkpoint/config edits, parameter pruning, target-driven tuning, and allocator rescue are not used. There is no second rescue candidate in this task.

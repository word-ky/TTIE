# T073B prospective native PromptIR memory schedule gate

2026-09-25 16:35 +08. This gate follows the earlier failed synthetic probes in `synthetic_4k_schedule_report.md`; it does not erase them. Status: **synthetic schedule accepted for one low-only execution smoke**, not accepted outputs or metrics.

## Fixed binding and scope

- Official DCTTA source `0526bf7b...`, five-task `epoch=80.ckpt` SHA256 `206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a`, float32, native 2160×3840, whole-image global attention retained. Both static PromptIR and PromptIR+DCTTA must use this same base.
- Schedule code SHA256: `chunk_conv_schedule.py` `1bd174c5c876e508ad1e49c78f3b6258c6d55acaae1dfd0dc819e78d436ddfe7`; `spill_forward_schedule.py` `985c2b55c7a9a1863b3f3c9ee7040b7ecf8bf949a453b02407788b3c3aa2ff39`. Synthetic probe SHA256: `probe_chunk_conv.py` `644df9f44fe1c644cc09611193a964472d907221348d2874d7da2b2bf623f0fd`.
- The schedule is applied only for full-image inference, not the official 320-patch DCTTA adaptation. It neither resizes nor tiles whole images, changes weights/normalization/attention, changes precision, nor reads target reference images.

## Why the execution is computation-equivalent

1. `qkv_dwconv` remains the original depthwise 3×3 `Conv2d`, on row windows with one-row halos; only interior rows are concatenated. All spatial neighborhoods and channel/weight mappings are unchanged. The global attention matrix and its spatial normalization are untouched.
2. Every FFN is the original `project_in` 1×1 → depthwise 3×3 → channel split → GELU/multiply → `project_out` 1×1, evaluated on 64-row halo windows and concatenated at interior rows. Each output row therefore sees the same pixels and operations as the official FFN; only intermediate allocation lifetime changes.
3. The official `PromptIR.forward` operation sequence is copied with the three encoder skip tensors moved losslessly to CPU RAM and back to the original CUDA device before each decoder concat. The original model modules, weights, dtype, and output addition are used. The copy is exact for float32 tensors.

This is mathematical/operator equivalence, **not bitwise floating-point identity**: CUDA selects shape-dependent convolution kernels. On checkpoint-loaded full-model synthetic comparisons, original versus scheduled maximum absolute difference was `0.00011942` (352 random), `0.00016892` (640 random), `0.00014848` (1024 gradient), `0.00021202` (640 dark), and `0.00014788` (1024 dark); all below the prespecified synthetic-probe assertion of `0.001`. Mean differences were `1.24e-5` to `4.91e-5`. At uint8 conversion, mismatch fractions were 0.00310 (352 random), 0.01250 (640 dark), and 0.00421 (1024 dark). Thus no claim of identical 8-bit outputs is made.

The independent native 2160×3840 *synthetic* runs in `synthetic_schedule_probes/4k_repeat1.json`, `4k_repeat2.json`, and `final.json` all produced finite float32 `[1,3,2160,3840]` output SHA256 `8f32ddc227d8df54faf36d9fceb5902fafc1364b491bc48f68d0b68259a5d922`, with peak reserved GPU memory `48,626,663,424` bytes on a 49,140-MiB card. The final receipt checked that all 548 model state entries were unchanged. The finite 4K synthetic run took 11.0 s. The margin is small, so any target OOM ends this single smoke without an unsealed alternative.

All receipts here have `reference_reads=0`, `target_model_runs=0`, `metrics=0`. No quality, superiority, or target-data claim follows from the synthetic runs. The next authorized action under the long-horizon B3b plan is **one** execution-only low-input smoke, then seal its geometry/dtype/finiteness/runtime/VRAM/hash/state receipt before any full output run. An OOM or failed integrity check returns B3b to an execution blocker; no automatic precision, resize, or tile rescue.

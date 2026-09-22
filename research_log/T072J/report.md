# T072-J — BLOCKED — clean-GPU native-4K baseline feasibility smoke

At `2026-09-22T04:52:10.764583842Z`, the pre-launch gate was checked on the A6000 host before opening the smoke payload or starting either model. The fixed requirement is one `NVIDIA RTX A6000` with at least `40960 MiB` free and no unrelated process above `1024 MiB`.

The host reported GPU 0 with `2078 MiB` free and GPU 1 with `3499 MiB` free. Both devices were occupied by unrelated `VLLM::Worker` processes using `44972 MiB`; the other recorded TTFL processes used `730 MiB` and `678 MiB`. Therefore no device qualified. The task is **BLOCKED** immediately. No process was killed, paused, evicted, or otherwise disturbed.

The receipt carries the exact frozen RetinexFormer and SNR-Aware bindings and the `1003_UHD_LL.JPG` identity sealed by T072-I (`3840×2160`, RGB, `752975` bytes, SHA256 `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`). T072-J did not open, stat, hash, decode, or read that payload. It also did not access GT/clean/reference data.

Both baseline runs remain `UNRUN`, Final Ours was not run, and all counters are zero: `inference_runs=0`, `model_launches=0`, `reference_reads=0`, `metrics=0`. No output geometry, output hash, runtime, or CUDA peak telemetry exists because the gate failed before launch. There was no retry or workaround.

`verify_receipt.py` independently checks the literal frozen bindings, the sealed smoke identity, the failed GPU gate, exact-one-run accounting, native geometry contract, and zero reference/input payload reads.

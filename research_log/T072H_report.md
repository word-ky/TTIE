# T072-H — BLOCKED: no qualifying clean A6000

The pre-inference GPU gate was checked at `2026-09-22T06:56:37+08:00` on the A6000 host. GPU0 had 3,497 MiB free and GPU1 had 3,499 MiB free. Each device was occupied by an unrelated VLLM worker using 44,972 MiB (`VLLM::Worker_TP0` PID 1337099 and `VLLM::Worker_TP1` PID 1337100). The fixed gate requires at least 40 GiB free and no unrelated process above 1 GiB, so neither device qualifies.

Per the stop rule, RetinexFormer and SNR-Aware were not launched, Final Ours was not rerun, no process was killed/paused/evicted, and the canonical `1003_UHD_LL.JPG` smoke input was not decoded. The frozen source/checkpoint/binding identities and smoke-input identity are recorded in `research_log/T072H/receipt.json`; the independent gate verifier confirms the block and zero-read ledger.

Accounting is `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `reference_reads=0`, `metrics=0`, and `smoke_input_decodes=0`. This is an environment-availability blocker and provides no native-4K feasibility conclusion for either baseline.

## Validation

`python research_log/T072H/verify_gate.py` passed with `classification=BLOCKED`, free memory `[3497, 3499]` MiB, `reference_reads=0`, and `metrics=0`. `python -m pytest -q research_log/T072H/test_gate.py` passed (`1 passed in 0.17s`). No scientific setting, checkpoint, config, precision, padding, resize, crop, tiling, normalization, or method code changed.

## Next step

Repeat this same bounded preflight only after one A6000 has at least 40 GiB free with no unrelated process above 1 GiB. Then run each frozen baseline once on the exact smoke input, preserving native geometry and the degraded-input-only read ledger.

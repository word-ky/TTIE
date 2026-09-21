# T072-F — BLOCKED: no qualifying clean A6000

The pre-inference clean-GPU gate was checked at `2026-09-22T04:24:25+08:00`. GPU0 had 2,832 MiB free and 100% utilization; unrelated VLLM worker PID 1337099 used 44,972 MiB and an unrelated TTFL Python process PID 2305513 used 660 MiB. GPU1 had 3,499 MiB free with VLLM worker PID 1337100 using 44,972 MiB. Neither device meets the required 40 GiB free and no unrelated process over 1 GiB.

Per the fixed stop rule, RetinexFormer and SNR-Aware were not launched, Final Ours was not rerun, no process was killed or evicted, and no UHD-LL low or gt/reference payload was decoded. This is an environment-availability blocker and provides no native-4K baseline feasibility conclusion. Real inference, optimizer, model-fit, reference-read, and metric counts are all zero.

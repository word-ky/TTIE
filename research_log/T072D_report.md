# T072-D — BLOCKED: no qualifying clean A6000

The clean-GPU gate was evaluated before any baseline inference. Both RTX A6000 devices were occupied by the unrelated vLLM service: GPU0 had 3,497 MiB free with `VLLM::Worker_TP0` PID 1337099 using about 44,972 MiB, and GPU1 had 3,499 MiB free with `VLLM::Worker_TP1` PID 1337100 using about 44,972 MiB. Neither satisfies the required 40 GiB free and at most 1 GiB unrelated-process gate.

RetinexFormer and SNR-Aware were therefore not run, no GPU process was killed or evicted, no image/reference was decoded, and no scientific setting or method was changed. The accepted T072-C Ours receipt remains the only reused evidence. This is an environment availability blocker; it does not establish baseline native-4K infeasibility.

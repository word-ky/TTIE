# T072-M — BLOCKED_GPU_GATE

The single authorized GPU gate was checked at 2026-09-22T17:22:55Z. GPU0 NVIDIA RTX A6000 had3495 MiB free; GPU1 had3497 MiB free. Both are below40960 MiB. Each GPU also had an unrelated VLLM worker using44974 MiB, exceeding1024 MiB. The raw device UUID/PID/process snapshot is in gate_raw.txt. No device qualifies.

Stopped before opening the canonical smoke input or launching a model. Retinexformer and SNR-Aware are UNRUN (zero executions each); Final Ours was not run. inference_runs=0, reference_reads=0, metrics=0, input_payload_reads=0, input_decodes=0, process_interventions=0. No further poll or retry occurred. No native4K feasibility result or model traceback exists because no model launched.

The receipt pins the declared T072-I input1003_UHD_LL.JPG, SHA256 cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca, native3840x2160 RGB and both exact accepted baseline binding digests. These are metadata bindings; no payload hash was recomputed in this blocked task.

Remote command (one SSH invocation through the existing workflow's Invoke-AutodlSsh, configured with this project's .autodl/config.json):

```
date -u +%Y-%m-%dT%H:%M:%SZ; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits && nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
```

This is one logical snapshot with one device query and one process query. Local script gate.ps1 preserves the command and endpoint configuration path. No unrelated process was disturbed. No training, data download or deployment occurred.

Independent verifier parses raw CSV and independently computes the failed gate, pins smoke/bindings and enforces zero-run accounting for this observed blocked branch. Synthetic tests reject nonzero counters, altered binding and a false blocked claim for a qualifying device. Reproduce with `python research_log/T072M/verify_receipt.py`. No output shape/finiteness/hash validation is claimed for unrun models.

Task authorized by origin/main cce84a1a. Stop after reporting; future GPU checks and full-cohort inference require a later task. Research-owned PROJECT_STATE.md unchanged.

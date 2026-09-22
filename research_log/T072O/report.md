# T072-O — NATIVE4K_SMOKE_LAUNCHER_SEALED

Sealed a future one-shot native UHD-LL baseline smoke launcher; this cycle used mocks only. real_inference_runs=0, real_input_payload_reads=0, reference_reads=0, real_metrics=0, process_interventions=0. No GPU query or remote command occurred.

## Exact invocation provenance

Accepted T071-B source579c3691a80f5b7cadfd706a2fe6750876c53aa0, research_log/T071B/run.py, reconstructs sys.argv and calls each exporter.main. The launcher preserves that same argv: --low CANONICAL_LOW --checkpoint ACCEPTED_CHECKPOINT --config ACCEPTED_CONFIG --out METHOD_OUTPUT; only SNR adds --source ACCEPTED_SOURCE. The canonical one-image list replaces the historical100-image list as authorized. The baseline_bindings.json file is copied byte-for-byte from that commit, preserving source/config/checkpoint/accepted-binding hashes and absolute checkpoint/source paths. spec.json includes immutable commits/blobs/SHA256 for both exporters, original wrapper and binding metadata. Independent verification reopens only those Git source blobs, not payload files.

Retinexformer remains default_no_gt_mean, no ensemble, float32/native RGB with existing reflect-pad4 and clamp semantics. SNR remains ttie_native_pad16 with its existing low-derived blur/SNR, reflect-pad16 and clamp. The unchanged exporters retain seed7 and deterministic cuDNN. The worker calls exporter.main instead of reimplementing either forward path. It adds only single-low decode restriction, exporter receipt checks and post-call peak-reserved telemetry. Existing synchronized runtime/peak-allocation values come directly from the exporters. The worker's final synchronize/max_memory_reserved query neither alters inference nor resets peaks.

## Contract

launcher.py first validates sealed metadata, then creates an exclusive output directory and takes one GPU/process snapshot. With no eligible NVIDIA RTX A6000 (free>=40960MiB and every other process<=1024MiB), it writes BLOCKED_GPU_GATE before binding/input/model access. Otherwise it selects the lowest eligible index deterministically, verifies all accepted asset hashes for both baselines before low access, verifies canonical input SHA/geometry, and executes Retinexformer then SNR in separate fresh processes on that one physical GPU. CUDA_VISIBLE_DEVICES limits DataParallel to that GPU.

There is no configurable input/reference path or inference-option override. The literal spec digest rejects any changed low path, declaration, threshold, binding or inference option. cv2 decoding inside the worker accepts only the canonical low; PIL decoding is denied during exporter execution as in T071-B. Each exporter writes its single output, and worker.py immediately hashes/verifies and freezes its telemetry before the next method. Unexpected OOM/runtime failure preserves captured stdout/stderr, traceback and started run count; execution terminates without retry. References and quality metrics have no stage in the launcher.

Spec SHA2564a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895. seal.json records SHA256 of launcher/worker/spec/provenance/tests/verifier/mock evidence and the aggregate root digest. Exact generated artifacts are authoritative; future changes invalidate the seal.

## Validation

Six focused synthetic tests pass: gate-before-access, binding-before-input, exact two-method success/order/telemetry, terminal failure on either method with no retry,11 rejected metadata mutations (including reference/GT/clean paths), and exact invocation reconstruction. Mock outputs use declared native metadata and hashes of synthetic strings; no real-size image or real model is created. Independent verify.py checks immutable Git provenance, file/root digests, mock event order, success fields/hashes, failure traces and zero real-cycle accounting. It imports neither launcher nor model code. PASS.

Reproduce locally: python research_log/T072O/test_launcher.py; python research_log/T072O/verify.py. Do not regenerate the sealed fixtures merely to verify them.

Future separately authorized command, from a runtime checkout containing the pinned T071-B sources/assets: `python research_log/T072O/launcher.py --runtime-root RUNTIME_CHECKOUT --out NEW_EXCLUSIVE_OUTPUT_DIRECTORY`. The runtime checkout must have the accepted relative config/binding paths and external assets at their pinned paths. The real backend and worker have not been executed in this cycle; the seal establishes mocked control flow and invocation provenance, not native4K feasibility. A failed prior output directory cannot be reused silently.

Research authorization09780a3a. PROJECT_STATE.md, dispatch and T072-L analysis spec unchanged. Stop after reporting; real smoke requires a later instruction and qualifying GPU.

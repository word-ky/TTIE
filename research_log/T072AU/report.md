# T072-AU — BLOCKED_GPU_AVAILABILITY

Authorization: origin/main 33a6ef0a, 2026-09-24 12:00 +08:00. Read-only resource-resolution task for the unchanged sealed native-4K objective. See `resource_evidence.md` for command outputs and configuration scope.

Current configured host `202.101.162.22:8220` exposes two NVIDIA RTX A6000s. GPU0/GPU1 have 3495/3497 MiB free. GPU0's VLLM worker PID 1337099 uses 44974 MiB; GPU1's VLLM worker PID 1337100 uses 44974 MiB. Both are root-owned, started 2026-09-20 02:05:33 +08:00, and share VLLM engine parent PID 1336825 (started 02:05:27). Neither GPU meets the frozen >=40960 MiB free and <=1024 MiB unrelated-process thresholds.

The existing workflow configuration identifies this one host and no alternate A6000 host; no legitimate clean execution window is exposed by the inspected metadata. This is a current-availability finding, not a prediction that the GPUs will remain occupied. No sealed launcher invocation, inference, input/reference access, metrics, allocation change, or process intervention occurred in T072-AU. Both baselines remain UNRUN and native-4K feasibility remains unresolved.

Classification: `BLOCKED_GPU_AVAILABILITY`. Scientific source, sealed launcher, baseline bindings, protocol, previous evidence, and `coordination/PROJECT_STATE.md` remain unchanged. Stop and await research-lead direction; no workaround or further attempt is authorized by this task.

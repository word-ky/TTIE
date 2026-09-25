# T072-AX — BLOCKED_GPU_AVAILABILITY

Authorization: origin/main 70619eac, 2026-09-24 15:58 +08:00 research-lead review. One fresh read-only GPU/process availability snapshot was taken on the configured authorized host at approximately 2026-09-24 16:12 +08:00 through the existing AutoDL workflow. The two `nvidia-smi` queries below form the single gate check; no later GPU check or launcher invocation was made.

```text
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits
0, GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, NVIDIA RTX A6000, 2296
1, GPU-9c468c54-b125-4903-1476-77c4d63270be, NVIDIA RTX A6000, 3497

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, 1337099, VLLM::Worker_TP0, 44974
GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, 3177322, /home/wenchang/asdasdsad/wjq/TTFL/.venv/bin/python, 1194
GPU-9c468c54-b125-4903-1476-77c4d63270be, 1337100, VLLM::Worker_TP1, 44974
```

GPU0/GPU1 free memory is exactly 2296/3497 MiB. The VLLM workers PID 1337099/1337100 use 44974/44974 MiB; GPU0 additionally has TTFL Python PID 3177322 using 1194 MiB. The added GPU0 process is a change from T072-AU through T072-AW, so a read-only `ps` metadata query was made: PID 3177322, parent 3177317, user shown as `liujian+`, started Thu Sep 24 16:08:17 2026, command is a Python `pprtp.run` job. No process was modified. Neither A6000 meets the frozen >=40960 MiB free-memory gate; both VLLM workers and the additional GPU0 process exceed the <=1024 MiB unrelated-process ceiling.

Classification: `BLOCKED_GPU_AVAILABILITY`. Sealed launcher invocations=0; inference runs=0; input/reference reads=0; metrics=0; process interventions=0. Retinexformer and SNR-Aware remain UNRUN, so native-4K feasibility remains unresolved. No scientific source/spec/binding/protocol change, no allocation or process change, and no `coordination/PROJECT_STATE.md` change. Stop; await research-lead direction.

# T072-AW — BLOCKED_GPU_AVAILABILITY

Authorization: origin/main a9f82410, 2026-09-24 14:57 +08:00 research-lead review. One fresh read-only availability snapshot was taken on the configured authorized host at approximately 2026-09-24 15:12 +08:00 through the existing AutoDL workflow. The two `nvidia-smi` queries below form this single availability check; no later GPU check or launcher invocation was made.

```text
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits
0, GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, NVIDIA RTX A6000, 3495
1, GPU-9c468c54-b125-4903-1476-77c4d63270be, NVIDIA RTX A6000, 3497

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, 1337099, VLLM::Worker_TP0, 44974
GPU-9c468c54-b125-4903-1476-77c4d63270be, 1337100, VLLM::Worker_TP1, 44974
```

GPU0/GPU1 free memory is exactly 3495/3497 MiB. The respective VLLM processes use 44974/44974 MiB. Neither A6000 meets the frozen >=40960 MiB free-memory requirement; both processes also exceed the <=1024 MiB unrelated-process ceiling. T072-AU/T072-AV had already established that this workflow configuration exposes no alternate authorized A6000; no deeper characterization was repeated.

Classification: `BLOCKED_GPU_AVAILABILITY`. Sealed launcher invocations=0; inference runs=0; input/reference reads=0; metrics=0; process interventions=0. Retinexformer and SNR-Aware remain UNRUN, so native-4K feasibility remains unresolved. No scientific source/spec/binding/protocol change, no allocation or process change, and no `coordination/PROJECT_STATE.md` change. Stop; await research-lead direction.

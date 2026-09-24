# T072-AU resource metadata — 2026-09-24 12:04–12:07 +08:00

Read-only inspection through the configured AutoDL workflow. The configured endpoint is user `liujianhua` at `202.101.162.22:8220`, remote base `/home/wenchang/asdasdsad/wjq/TTIE`. The project workflow config has one `host` field and no alternate host/device list. No other infrastructure authorization was found in that configuration.

The workflow status command reported two NVIDIA RTX A6000 devices, each with 49140 MiB total and approximately 45045/45043 MiB used. Its final running-process section hit a transient SSH timeout; the direct read-only queries below subsequently succeeded. No process was modified.

```text
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits
0, GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, NVIDIA RTX A6000, 3495
1, GPU-9c468c54-b125-4903-1476-77c4d63270be, NVIDIA RTX A6000, 3497

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe, 1337099, VLLM::Worker_TP0, 44974
GPU-9c468c54-b125-4903-1476-77c4d63270be, 1337100, VLLM::Worker_TP1, 44974

ps -p 1337099,1337100 -o pid,user,lstart,etime,args --no-headers
1337099 root     Sun Sep 20 02:05:33 2026  4-10:00:48 VLLM::Worker_TP0
1337100 root     Sun Sep 20 02:05:33 2026  4-10:00:48 VLLM::Worker_TP1

ps -p 1337099,1337100 -o pid,ppid,user,args --no-headers
1337099 1336825 root     VLLM::Worker_TP0
1337100 1336825 root     VLLM::Worker_TP1

ps -p 1336825 -o pid,ppid,user,lstart,args --no-headers
1336825 1335709 root     Sun Sep 20 02:05:27 2026 VLLM::EngineCore
```

Both authorized GPUs fail the unchanged >=40960 MiB free-memory gate and <=1024 MiB unrelated-process gate. The two workers share a VLLM engine parent and have existed since 2026-09-20 02:05 +08:00 according to `ps`. This metadata does not reveal a clean future reservation/window or authorize any intervention. No qualifying device was found in the configured host, and no alternate authorized host is exposed by the existing workflow config.

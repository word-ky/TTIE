# T072-N — GPU_BLOCKER_PROVENANCE_CHARACTERIZED

One read-only snapshot completed at2026-09-22T18:08:39.064578+00:00. Both original blockers remain running as OS user root. PID1337099 and PID1337100 share PPID1336825; their ps comm is VLLM::Worker_TP (OS-truncated), while nvidia-smi identifies VLLM::Worker_TP0/TP1. Both started at host-local `Sun Sep20 02:05:33 2026` and have elapsed259385seconds (72h03m05s). The host-local timestamp is preserved without assuming a timezone.

Both four-level parent chains are identical:1336825 VLLM::EngineCor ->1335709 vllm ->1335687 containerd-shim ->1 systemd. Together with the matching PIDs across prior observations and72-hour uptime, this supports interpreting them as persistent containerized VLLM jobs. Their service purpose, owner project, expected finish time and release schedule are not established by these allowed metadata. Both cwd readlink operations returned PermissionError errno13; no privilege escalation or alternate probing was attempted. User/start/elapsed/parent identity were available, so optional cwd denial does not trigger BLOCKED_PROVENANCE_PERMISSION.

Exactly two GPUs are visible, both NVIDIA RTX A6000. UUID GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe (GPU0) has3495MiB free; UUID GPU-9c468c54-b125-4903-1476-77c4d63270be (GPU1) has3497MiB free. Each worker uses44974MiB. Each device fails the unchanged >=40960MiB free and <=1024MiB per unrelated-process requirements. No additional A6000 is visible and no device qualifies.

Scheduling implication: another immediate identical retry has little expected value. A later resource-allocation decision or availability change is needed before a separately authorized smoke. No inference or further polling was launched.

Exact command arrays, return codes and sanitized outputs are in snapshot.json. snapshot.py collects one nvidia-smi GPU inventory, one compute-process inventory, ps user/PID/PPID/comm/lstart/etimes for each blocker, at most four ps PID/PPID/comm parent levels, and os.readlink on each /proc/PID/cwd. It does not read command arguments, environ, history, credentials, open-file contents or memory. snapshot.ps1 records the existing workflow helper and project-specific connection configuration. No workload or files were deployed remotely; Python executed the collector in memory once.

process_interventions=0, inference_runs=0, input_payload_reads=0, reference_reads=0, metrics=0. No target input, reference or metric was accessed. Lightweight local verify.py replays raw CSV/ps output, device gate, parent chains and zero accounting; PASS. It touches no host and opens no target data. Run `python research_log/T072N/verify.py` locally to regenerate receipt.json.

Authorization7c0af20b. PROJECT_STATE.md unchanged. Stop after reporting; no native4K smoke retry under T072-N.

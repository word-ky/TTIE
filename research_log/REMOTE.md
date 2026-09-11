# TTIE A6000 and heartbeat recovery

- Local project: `D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement`.
- Remote host verified as `wenchang-PR4904W1`, two NVIDIA RTX A6000 48GB GPUs.
- Existing writable wjq found at `/home/wenchang/asdasdsad/wjq`; new isolated project `/home/wenchang/asdasdsad/wjq/TTIE` created on 2026-09-11 UTC (September 12 Asia/Shanghai).
- Project venv: `/home/wenchang/asdasdsad/wjq/TTIE/.venv`. Base interpreter: `/home/wenchang/anaconda3/envs/python3.12-tk2-2.3/bin/python`, system-site-packages enabled. Existing PyTorch reused; Pillow installed only in the TTIE venv. Default remote shell has no `python` in PATH.
- Workflow root: `D:/work/claude-autodl/autodl-workflow-clean`. Use its scripts with `AUTODL_CONFIG_PATH` pointing to this project's ignored `.autodl/config.json`. Do not use the workflow's default remote base (an unrelated project).
- Deploy: `scripts/autodl-deploy.ps1 -Tag ttie-t001 -Source <local-project>`.
- Run: `scripts/autodl-run.ps1 -Name ttie-t001-a6000 -Cmd 'bash scripts/run_a6000.sh'`. Script runs CPU tests, CPU toy, two deterministic CUDA toy runs. Each is only 200 updates per mode, no external data or models.
- Inspect with `scripts/autodl-logs.ps1 -RunId <id> -Lines 80`; copy full receipts into this project's `research_log/remote_runs/<id>` via existing `Copy-FromAutodl` helper. Keep explicit run/release IDs in project log because workflow global last-run state is shared with other projects.
- Remote `releases/`, `current`, `runs/`, `shared/` belong solely to TTIE. No other project environment or jobs should be changed.

Heartbeat `ttie-chatgpt` is ACTIVE, every 15 minutes, attached to this Codex task. It reads main's coordination inbox/state and PR feedback, executes authorized OPEN work or requested revisions, and sends results through the Codex-owned GitHub mailbox. ChatGPT's hourly review is user-reported; an actual future two-way scheduled exchange has not yet been observed. Do not rerun a completed T001 merely because the research lead has not changed OPEN to DONE. Read T001.md and the latest mailbox report on resume. All durable state belongs in this project.

Completed release: `20260912-010250-ttie-t001`, source code commit `17f6563f5d6aa0532aa8fab3cc6023d6f7195b4b`. Completed run: `20260912-010311-ttie-t001-a6000`, exit 0 at 2026-09-11T17:03:45Z. Receipts exist remotely under `runs/<id>` and locally under `research_log/remote_runs/<id>`. No TTIE job remains active after completion. Tests: 13/13 CPU; repeated CUDA outputs exactly equal. Research lead acceptance pending.

## Latest state — T002 completed, 2026-09-11 UTC

T001 has been accepted and merged. The scheduled Codex heartbeat received ChatGPT's T001 acceptance and T002 task at 17:54:54Z, so GitHub-mediated two-way communication is now observed (supersedes earlier unobserved notes).

Latest tested code `052a4361df983da812213c853459b1cb84f4558c`, release `20260912-020216-ttie-t002`, run `20260912-020232-ttie-t002-a6000`, command `bash scripts/run_t002_a6000.sh`. Finished at 2026-09-11T18:23:14Z with exit 0. Twenty CPU tests, frozen T001 regression, all-model CPU/CUDA parity and exact CUDA repeat passed; full fixed CUDA matrix contains 504 rows, all finite. Full receipts copied to local `research_log/remote_runs/<id>`. No TTIE job remains running. Research lead reviews T002 before new work. Remote current still points to the tested T002 release; updated recovery notes are mirrored separately under the remote project root research_log.

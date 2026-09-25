# T072-AY resumed host preflight — 2026-09-24 17:54 +08:00

User explicitly supplied access to a new paid GPU host after the earlier T072-AY secure-handoff completion. A private, ignored local workflow configuration was created and the existing local SSH public key was authorized on that host through an interactive SSH password prompt. The password was not placed in a command argument, repository file, PR, or coordination record. Subsequent key-based AutoDL workflow access succeeded. No password or token is included here.

One remote GPU/process preflight showed one device: `NVIDIA GeForce RTX 4090`, UUID `GPU-5b9516cb-131f-1862-ae47-1e7ff15207ce`, 49140 MiB total, 48510 MiB free; the compute-app query listed no processes. Thus the free-memory and unrelated-process portions of the T072 gate pass.

The exact frozen T072-O launcher cannot select this device: `research_log/T072O/launcher.py` line 24 requires `row['name']=='NVIDIA RTX A6000'`, and sealed `spec.json` declares that same name. Running it unchanged on the new RTX 4090 would return `BLOCKED_GPU_GATE` before bindings or input; silently relaxing the name check would change the sealed source/spec. T072-AY explicitly prohibits source/spec/binding edits. The sealed spec and accepted bindings also retain absolute paths from the prior host; relocation would require an approved, independently verified portability plan.

Stopped at this first genuine contract blocker. No code/checkpoint/cohort staging, sealed launcher invocation, model inference, output freeze, reference read, metric, or scientific-state change occurred on the new host. Prior T072-AY secure-handoff completion remains historically accurate for its earlier attempt; this entry records the later user-authorized resumption.

Decision needed: authorize a narrowly scoped re-seal/port of only the host-specific GPU-name and absolute-path bindings for the new device while preserving checkpoint/source/cohort hashes and native-4K float32 behavior, or provide an A6000 host that satisfies the original sealed launcher unchanged. Do not launch baseline inference until that choice is explicit.

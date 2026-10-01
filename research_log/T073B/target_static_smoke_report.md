# T073B PromptIR native target-low static smoke

2026-09-25 16:45 +08. **Execution-only smoke PASSED; full row still pending.** The prospectively published synthetic schedule gate is `native_schedule_gate_20260925.md`, GitHub branch commit `034fd249e8b80b58a9d23fc7ca9cde1725a9e82d` before this target attempt.

The single canonical low input was `1003_UHD_LL.JPG`, SHA256 `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`, matching frozen T072-I. The task-owned low-only loader did not accept a reference directory. Official five-task checkpoint SHA256 `206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a`. Smoke script SHA256 `e3d6597833894cb44987958380440f2f36a5795d8601cc5e1b5573826c8efbc4` matches the remote file.

Output was finite float32 `[1,3,2160,3840]`, raw tensor SHA256 `edd0d508e737f5460cb93e2cc153479b3f72e9948ab9ebb19d75ca1c1d00db2b`. Forward time 11.027 s, peak CUDA reserved 49,148,854,272 bytes; all 548 model state keys unchanged. The schedule bound 94 depthwise convolution modules and 47 FFN modules. Machine receipt `target_static_schedule_smoke.json` SHA256 `935101dedbe219194f6eda3e9d7103ba9eabc4ec2114f744ad35cbad775e98f6`.

No output quality or model ranking was examined. `reference_reads=0`, `metrics=0`. This one successful smoke allows a prospective full **static** PromptIR low-only 150-image run with the *same* source, checkpoint, schedule, dtype, and native geometry, followed by independent output verification. It does **not** establish the DCTTA full 4K adaptation/inference path, freeze either row, or unlock references.

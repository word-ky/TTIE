

---

## T072-AY — BLOCKED_SECURE_CREDENTIAL_HANDOFF

Evidence eb74345cf30434085996989893e61774b36f0c6d; PR https://github.com/word-ky/TTIE/pull/218.

The newly user-authorized paid GPU host was not available through a secure execution channel in the local AutoDL workflow. The existing workflow configuration remains on the blocked old host; no matching usable key-based new-host connection was configured. Credentials exist only out-of-band in another conversation and were not copied into the repository or captured shell commands. A direct remote-terminal attempt in that conversation was blocked by its security layer; no bypass was attempted.

New-host GPU snapshots=0, staging=0, sealed launcher invocations=0, baseline inference=0, outputs frozen=0, reference reads=0, metrics=0. Both UHD-LL baselines remain UNRUN. No scientific or PROJECT_STATE change. Non-secret handoff needed: configure authorized private key-based new-host access for the local workflow and separately authorize resumption. Stopped; do not fall back to old host.


---

## T072-AY re-entry diagnostic — sealed GPU model gate

Evidence 9e2bc77ddce4357b8ba1f5f2cb92e9a5d0ff6a0c; PR https://github.com/word-ky/TTIE/pull/219. This is a non-completion addendum to the earlier secure-handoff report. User-provided access to the new paid host was established securely with private key-based workflow access; no credential is in the repository or this note. One preflight found one NVIDIA GeForce RTX 4090 with 49140 MiB total, 48510 MiB free, and no listed compute processes. Free-memory/process thresholds pass, but the unchanged sealed T072-O launcher requires the exact device name NVIDIA RTX A6000, so it cannot select this host. Original sealed spec also contains prior-host absolute paths. No code/data staging, launcher invocation, inference, reference read, metric, or PROJECT_STATE change occurred. Explicit research-lead/user authorization is needed for a minimal new-host re-seal/port, or an original-spec A6000 host.

---

## T072-AZ — BLOCKED_NATIVE4K_SNR_AWARE

The paid RTX 4090 host passed the GPU gate (48510 MiB free, no listed compute process). The minimal device/path re-seal passed independent verification of all 70 frozen source/checkpoint/config assets and all 150 canonical T072-I low-only inputs; no scientific bytes changed. Exactly one Retinexformer native-4K float32 smoke passed (finite output, 2.286 s, peak allocated 20.35 GB). Exactly one SNR-Aware native-4K float32 smoke failed with genuine `torch.cuda.OutOfMemoryError` in transformer attention mask application, attempting to allocate 31.29 GiB. The stop condition fired immediately: no second smoke, no workaround, neither 150-image full run started, zero full outputs, `reference_reads=0`, `metrics=0`, and no PROJECT_STATE change. Raw GPU snapshot, port manifest/verifier, source/input hashes, smoke receipts, traces, and report are in `research_log/T072AZ/`. Research-lead decision is needed before any revised SNR-Aware protocol or hardware attempt.

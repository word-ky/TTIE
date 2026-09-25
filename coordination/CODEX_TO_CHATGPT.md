

---

## T072-AY — BLOCKED_SECURE_CREDENTIAL_HANDOFF

Evidence eb74345cf30434085996989893e61774b36f0c6d; PR https://github.com/word-ky/TTIE/pull/218.

The newly user-authorized paid GPU host was not available through a secure execution channel in the local AutoDL workflow. The existing workflow configuration remains on the blocked old host; no matching usable key-based new-host connection was configured. Credentials exist only out-of-band in another conversation and were not copied into the repository or captured shell commands. A direct remote-terminal attempt in that conversation was blocked by its security layer; no bypass was attempted.

New-host GPU snapshots=0, staging=0, sealed launcher invocations=0, baseline inference=0, outputs frozen=0, reference reads=0, metrics=0. Both UHD-LL baselines remain UNRUN. No scientific or PROJECT_STATE change. Non-secret handoff needed: configure authorized private key-based new-host access for the local workflow and separately authorize resumption. Stopped; do not fall back to old host.


---

## T072-AY re-entry diagnostic — sealed GPU model gate

Evidence 9e2bc77ddce4357b8ba1f5f2cb92e9a5d0ff6a0c; PR https://github.com/word-ky/TTIE/pull/219. This is a non-completion addendum to the earlier secure-handoff report. User-provided access to the new paid host was established securely with private key-based workflow access; no credential is in the repository or this note. One preflight found one NVIDIA GeForce RTX 4090 with 49140 MiB total, 48510 MiB free, and no listed compute processes. Free-memory/process thresholds pass, but the unchanged sealed T072-O launcher requires the exact device name NVIDIA RTX A6000, so it cannot select this host. Original sealed spec also contains prior-host absolute paths. No code/data staging, launcher invocation, inference, reference read, metric, or PROJECT_STATE change occurred. Explicit research-lead/user authorization is needed for a minimal new-host re-seal/port, or an original-spec A6000 host.

---

## T073-A — UHDLL_EXPANDED_MAIN_TABLE_PREREG_SEALED

Accepted T072-AZ-B output manifest from PR #220 was verified at SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`. The prospective ten-row registry and metric plan under `research_log/T073A/` preserve T072-L's exact T071-B RGB metric provenance and one shared PCG64 seed `20260922` / 10,000-resample paired stream, with immutable comparison signs, strict-positive wins, complete 150-image policy, and reference gate. Only RetinexFormer/SNR-Aware are `FROZEN_OUTPUTS`; eight rows remain `PENDING_OUTPUTS` and require exact prospective source/artifact/protocol bindings before execution. Independent verifier passed Git-object SHA256 checks and 11 fail-closed mutations; task counters `reference_reads=0`, `metrics=0`, `model_runs=0`. No PromptIR/DCTTA or other pending method was executed. UHD-LL references remain sealed until all final Tier-1 outputs are independently frozen and ZERO-IG/GM-MoE inclusion is locked; next task awaits the research lead.

---

## T073-A-R1 — corrected preregistration, research-lead acceptance pending

The original T073-A seal claim above is superseded by R1 review at `origin/main` `cbd79646`. Revised PR #221 fixes exactly eight Tier-1 methods (ZERO-IG preferred additional, GM-MoE secondary), holds PromptIR and PromptIR+DCTTA at a shared unresolved `PENDING_SOURCE_BINDING`, and requires `LOW_ONLY_DCTTA_REQUIRED` with an independently verified low-only wrapper/equivalence before model execution. T072-L provenance, complete 150-image policy, seed `20260922`/10,000 shared resamples, comparison conventions and T072-AZ-B manifest SHA256 are unchanged. Independent verifier passes with 17 rejected mutations; `reference_reads=0`, `metrics=0`, `model_runs=0`. `PROJECT_STATE.md` now says R1 locally verified, not lead-accepted. No pending method was run; stop here and review R1 before authorizing PromptIR/DCTTA execution.

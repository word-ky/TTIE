# T061-B — PASS: source-only stopping step frozen

The immutable source-only rule selects **k* = 11** (unique maximum; tie set [11]). This is a source-side candidate, not evidence of development transfer or a promoted method.

- Authorization: `43d7a04fc22a74b84ef3029cd9c01197d7c0aaa1`.
- Rule preregistration: `5da5a57e0b481a98d5087aacf7afde8771f80217`; rule SHA-256 `9242c8948aab87eeb1fe0a9bc59e61e6b4de33c06d50014cedad74ace8fc9c12`.
- Tested source: `d5eed1e2419e296b566caab3d82427eb8796d14c`; selector SHA-256 `33ca7e5bd36130d24776d3706b4bbf11eea304ea9097c9071d3454c14c5ff87c`.
- Accepted source evidence: `23c98d16c159e8f51c5147647281ae749741dade`. The exact Git blob IDs and SHA-256s for both files are in `result/manifest.json` and `choose.INPUTS`.
- Freeze UTC: **2026-09-19T18:17:49.524356+00:00**; manifest SHA-256 **`34a15a13f0c94b07a7eef870efee9f28da51e30abc7f440a342a62da5c0cd1b5`**.
- Source mean PSNR at step 11: **17.566028939521 dB**. All 41 means are preserved in step order in the manifest.

Validation: all 60 unique source anchors, matched to the accepted freeze in exact order, contain 41 finite method-A PSNR values (2,460 total). Hashes and accepted Git blobs match. Three synthetic tests passed (0.281 s), covering unweighted arithmetic/earliest ties, required invalid-input rejection, and attempted non-allow-listed file access. One real selector invocation completed successfully. A separate verifier independently assembled columns and summed exact binary64 inputs with 80-digit Decimal arithmetic: k* and tie set agree exactly; maximum mean difference is 3.5527136788005009e-15 dB. No failed real invocation or rerun.

Runtime data reads were restricted to accepted T060-D-R2 `metrics.json` and `freeze.json`, plus the newly created output directory. Python/script imports complete before the audit hook is installed and before data is opened. The accepted metrics container includes both A and B, but only A PSNR enters arithmetic; no T059-E model is used. The runtime read lists are preserved. The hook denies file opens outside the explicit allow-list; this is a Python-level restriction, not an OS sandbox.

The T061-A historical development-summary exposure is retained and disclosed; this task makes no renewed blindness claim. The rule was fixed before that exposure and was not modified. This task read no T037/LOL-v2 development quality summary, target data, official test or cross-dataset data. No new optimizer, image/reference decoding, learned selector, or development evaluation occurred. Standard-library CPU analysis is sufficient; no GPU computation was needed.

Commands, script hashes and stdout are in `commands.json`, `selector_run.log`, and `verifier_run.log`. To reproduce in a fresh output directory, use the recorded commands with local paths adjusted and the same accepted input bytes. Existing output files cannot be overwritten by the selector.

Next: research-lead review of the immutable candidate. Development transfer remains deliberately unevaluated; do not execute unchanged OPEN T061-B again or extend it into evaluation without a new task.

# Long-horizon objective — Ours-v2 development SOTA

Date: 2026-10-01 (Asia/Shanghai)

The project owner renewed the long-horizon objective: continue developing and
validating the Ours-v2 development version until it reaches a verifiable SOTA
result on the declared target datasets and comparison main table, then stop
only after all required tasks are complete.

Execution constraints:

- every reported number must come from an actual reproducible run;
- test-GT-tuned ceiling results must remain explicitly labeled as such and
  must not be presented as held-out generalization;
- sealed references, low-light-only scope, and declared method/evaluation
  boundaries remain unchanged;
- no metric, target, baseline, or output may be fabricated or altered to
  manufacture SOTA;
- each material step, artifact hash, and verified result is written to the
  project log and pushed to GitHub;
- on G4, use only physical GPUs 0 and 2 and never touch unrelated jobs on
  GPUs 1, 3, 5, or 7;
- paid GPU work must be finite and purposeful; completed work is not repeated.

Current execution handoff: finish the active SID ceiling pipeline on GPU0,
then continue the declared LSRW development search/verification on GPU2 as
appropriate, and update the comparison table only after independent
verification. The existing thread-level long-horizon goal remains the single
goal; this file records its renewed success condition rather than creating a
parallel goal.

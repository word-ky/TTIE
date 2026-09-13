# T018-D PR28 provenance review fix

## 2026-09-13T04:50:20.708573+00:00 — reproduced and patched

Review comment 3998734227 (automated reviewer, reviewed d75461f6) reports that the verifier cannot read T016-B inputs from a clean checkout without training-workspace Git objects. Main unchanged 0b0ef7b3663d9faffbcafec7e8eaf3d35c7ae649; no new research task. PR28 head 6f1f9e0eda038628357a8cd0cd627c5523341121.

Reproduced using `git clone --no-local --single-branch --branch codex/T018D-final-direction-selector . .autodl/T018D-clean-review`. This transfers only reachable branch objects rather than copying the training Git object store. Source commit is present; scoring commit 4062e01c is absent. With core.autocrlf=false and original committed bytes restored, the original verifier exits1 on git-show of selection_receipt.json. See T018D_review_before.json/log. Initial Windows checkout used global line-ending conversion and failed an earlier source-byte assertion; that setup log is retained separately. The test clone now preserves exact bytes.

Minimal repair: read the already committed six T018D_source_inputs copies, require manifest entries to equal the immutable receipt's original commit/path/hash records, and require each payload SHA256 to match. No fallback, refetch, new model, changed target values, or weakened check. Original source-byte checks and target-free independent replay are retained. No changes to scientific source files, model weights, normalization, predictions, or pinned selector receipt.

Next: commit the verifier fix, fast-forward the isolated clone, run its independent verifier with the historical scoring object still absent, deliver the review reply and main outbox, and sync recovery notes to both server roots. No retraining or GPU experiment is needed for this verifier-only fix.

## 2026-09-13T04:51:15.951669+00:00 — preserve committed payload bytes

The first patched clone run exposed a separate Windows checkout detail: three archived score files remained CRLF-converted on disk, although their committed Git blobs exactly match the frozen SHA256. Read the committed copies directly with git show HEAD:research_log/T018D_source_inputs/<name>.json. This single path both avoids unreachable historical input objects and preserves original bytes under checkout conversion; no hash normalization or tolerance. Source and receipt checks remain unchanged. Archived-checkout failure log preserved.

## 2026-09-13T04:51:53.922799+00:00 — isolated regression passed

Clean reachable-object clone at 9ef55c70e46defff02a04611c81108433f284c69 still lacks scoring commit4062e01c. Revised verifier exits0; all9 source hashes,6 input hashes,8 frozen artifact hashes pass. Independent target-free subprocess exactly reproduces120 logits/classes/decisions and both normalization buffers. Pinned receipt db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94 unchanged. See T018D_review_after.json/log and T018D_review_verification.json. No retraining or broader scientific regression needed for this verifier-only change.

## 2026-09-13T04:53:44.202717+00:00 — concurrent squash merge

Research lead merged the original PR28 head6f1f9e0e as9e3a2447 while this repair was being prepared, so the repair commits are not in main. Transferred the bounded patch to codex/T018D-verifier-repro based on updated main0aa6d639. The squash also makes historical scientific-source commit6c6eda3a unreachable in a fresh main clone. Source verification now compares current committed scientific blobs and runtime bytes with the unchanged pinned receipt hashes; original source SHA remains recorded in receipt/config. Thus both scientific code and input payloads are verified without requiring unreachable pre-squash objects. New OPEN taskT018-E recorded for subsequent work, after this patch delivery.

## 2026-09-13T04:55:07.321327+00:00 — post-squash shallow clone verified

A fresh depth1 checkout391fb87e contains neither historical scientific-source6c6eda3a nor scoring4062e01c object. Verifier exits0 and all120 reference-free replay decisions match exactly; source/input/final hashes pass. See T018D_review_squash.json/log. Repair complete; separate follow-up PR required because original PR28 merged before repair. T018-E is now authorized on main0aa6d639.

Review fix delivered in PR29; main outbox ebeb6cb0ed364a7d3f31f811aaf3449250b27652. Recovery patch accompanies the prior archive. Continue newly issued T018-E separately.

Both remote archive/verifier hashes verified against local. Review repair delivered, no further work in this branch.

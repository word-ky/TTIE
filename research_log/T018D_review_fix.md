# T018-D PR28 provenance review fix

## 2026-09-13T04:50:20.708573+00:00 — reproduced and patched

Review comment 3998734227 (automated reviewer, reviewed d75461f6) reports that the verifier cannot read T016-B inputs from a clean checkout without training-workspace Git objects. Main unchanged 0b0ef7b3663d9faffbcafec7e8eaf3d35c7ae649; no new research task. PR28 head 6f1f9e0eda038628357a8cd0cd627c5523341121.

Reproduced using `git clone --no-local --single-branch --branch codex/T018D-final-direction-selector . .autodl/T018D-clean-review`. This transfers only reachable branch objects rather than copying the training Git object store. Source commit is present; scoring commit 4062e01c is absent. With core.autocrlf=false and original committed bytes restored, the original verifier exits1 on git-show of selection_receipt.json. See T018D_review_before.json/log. Initial Windows checkout used global line-ending conversion and failed an earlier source-byte assertion; that setup log is retained separately. The test clone now preserves exact bytes.

Minimal repair: read the already committed six T018D_source_inputs copies, require manifest entries to equal the immutable receipt's original commit/path/hash records, and require each payload SHA256 to match. No fallback, refetch, new model, changed target values, or weakened check. Original source-byte checks and target-free independent replay are retained. No changes to scientific source files, model weights, normalization, predictions, or pinned selector receipt.

Next: commit the verifier fix, fast-forward the isolated clone, run its independent verifier with the historical scoring object still absent, deliver the review reply and main outbox, and sync recovery notes to both server roots. No retraining or GPU experiment is needed for this verifier-only fix.

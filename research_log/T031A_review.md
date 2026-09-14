# T031-A PR #56 review follow-up

Both P2 comments from 2026-09-14T12:10:03Z are addressed. The reference diagnostic now requires `--deployment research_log/T031A_deployment.json` and checks its separately persisted `support_freeze_sha256` before reading bound tensors, loading models or installing the image opener. The output receipt records the checked hash and deployment-file hash. A replaced support freeze fails this check. This repairs the missing executable binding; it does not retroactively claim the original GPU run executed the new check.

The existing deployment receipt and actual frozen support directory agree on `3b84948baed0c84e257365a2c9ed8be18148b853327778dc78dc96e9a4037756`. Original run commands/logs/receipts remain unchanged. To execute the revised diagnostic, add the required `--deployment` argument to its saved command and use a new output path; no new diagnostic was run for this review.

The saved-tensor exporter now creates its output parent directory. The documented export/replay pair was executed with a previously absent `research_log/T031A_download/review-fresh/` destination and completed successfully. All 4,100 distances and gradient scalar summaries remain verified, maximum distance error `4.440892098500626e-16`, AUROC `0.7204698309323111`, rho `-0.48229925365892745`.

Validation: `python -m pytest -q tests/test_t031a_support.py` — **4 passed in 27.63s**, including accepted/replaced external freeze binding and pre-opener check ordering. Fresh-directory export and independent saved-evidence replay both exit 0. These are CPU tooling checks; no GPU experiment, image evaluation, optimizer update or selection was repeated. Classification remains **promising source-support proxy**. Await research-lead review; no self-merge.



---

## T021-A — PARTIAL — baseline/pool clarification required before scoring

UTC: 2026-09-13T16:09:44.169634+00:00

Draft PR: https://github.com/word-ky/TTIE/pull/41 ; branch `codex/T021A-ssim-audit`; preparation commit `694cd5bd248f3b938660e88e12ca25f6876b495d`.

Exact accepted T014 Stage-B artifact recovery succeeds. PR14 accepted merge `cebecffbd1335fade336df17d653eb4e5fb65ba3` binds the manifest/config and historical output verification. All480 episode output/decision files across240 rows/40 source images match original SHA256 and sizes (3924677163 bytes). Verification reads bytes only: no tensors or clean-reference pixels loaded. Stored clean-condition identity tensors provide the original same-image clean reference linkage without regeneration or resizing. No TTT or candidate rerun is needed. This is not a structurally unsupported result.

**Please bind two scientific choices in the inbox before evaluation:**

1. Which exact stored method is H0? T014 has identity, region2_direct, region2_discrete_projected, fixed_step_source (16), matched region2_ttt_energy_value_only and other controls, with different comparisons in its accepted gates. There is no unique per-row H0 alias in the accepted T014 source/receipts. Naming only “the exact accepted baseline” does not identify a single comparison.
2. Which rows define the pooled bootstrap? The original fresh run contains200 primary rows (five conditions),80 heterogeneous qualification rows (left_right/quadrants subset), and40 additional offset rows explicitly report-only. Specify200 primary,80 heterogeneous, or240 including offset. I have not inspected SSIM outcomes to choose a favorable pool.

Completed code: fixed float64 full-RGB Gaussian11×11/sigma1.5 SSIM, population statistics, K1=.01/K2=.03, symmetric reflection padding, no crop/resize/normalization. Cross-check compares the complete skimage0.24 SSIM map, since its scalar omits the border and the task requests no crop. Fixed10000 image-cluster bootstrap uses seed7, NumPy PCG64, all rows/multiplicity per sampled image, percentile95%/linear quantiles. `python -m pytest tests/test_ssim_transfer.py -q` =>2 passed1.01s, including full-map/edge cross-check and unequal-cluster row-count test. Tests use synthetic arrays only.

Files: `ttie/ssim_transfer.py`, focused tests, `research_log/T021A_inputs/`, `T021A_verify_available_outputs.py`, `T021A_outputs_verified.json`, `T021A_preparation.md`, progress and test receipts. No real per-row SSIM, bootstrap CI or scientific verdict has been computed. No engineering failure or change to outputs/decisions. Await the two bindings; then continue the same task/PR without repeating recovery. Research-owned files remain untouched.

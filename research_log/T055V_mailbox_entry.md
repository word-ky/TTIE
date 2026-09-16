

---

# T055-V DONE — T055 replay verified — 2026-09-16T16:13:59.201456+00:00

Verifier source `49606b5d654a5e69b68b9895e81759c0f01cd175`; evidence `70041a563410e9bb7580d77357d02cb4403c4b66`; [PR #81](https://github.com/word-ky/TTIE/pull/81). Frozen T055 source `e6a79d80740307fbea3b4391a9e9f9a718f14e89` / evidence `202de51182e50d5cc40e19b65992106487175aba`, accepted T054 `f4baa579` / `a4d37006`, cohort `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b` unchanged.

Diagnosis: index12 pixel(200,40), old NumPy raw −.0975489616394 versus CUDA −.0975500643253; coefficient error1.0952353477478027e-6. CUDA reconstruction from saved rawv alone exactly matches savedc. A separate verifier now emulates CUDA fused multiply-add at both horizontal weighted sums and the vertical sum; no image/value/shape/outcome special cases. Corrected raw interpolation matches CUDA exactly; coefficient error1.1920928955078125e-7. Nine independent synthetic cases across three shapes also give rawerror0. Before/after diff and exact raw controls/locations are in `research_log/T055V_report.md` and `T055V/verifier.diff`.

Full replay `20260917-000911-ttie-t055v-replay` PASS, exit0: **100images,100100history states,200metrics,724scalarchecks**. Max errors: basis1.771841198205948e-7, interpolation1.1920928955078125e-7, renderer0, scalar3.552713678800501e-15. Unchanged-state/inactive/order/aggregate/near-boundary checks pass. Before/after299file manifest, including86 original source bindings, is identical (manifest SHA514172ee1dd73e2635b5a045b40bebf63fd1dafe6ced7c9b6eaf336a325e2709). Old verifier SHA95ee4a247c4830bcc5cfae704de5c3dd5dc49cb31cad54d44b2d059be3aa6e64 preserved; separate new verifier SHAac5a055aab5f8bf9380fd3587841589e361d52d57dfeed3f9ecb515e7396cd69. No optimizer, new states/outputs, metric-file regeneration, or tolerance/settings/gate changes.

**T055 replay verified. Final original verdict: material local-detail underconvergence not supported under fixed extension.** Unchanged paired meanPSNR+.004305474682432848, median+.002918943444738531, meanSSIM+.0013706655157820719 miss .25/.10/.010 gate. Close pure step/LR-budget rescue for this one-scale family; no global-convergence claim or follow-on experiment.

Failures: initial CRLF copy-replacement miss caused synthetic tests to exercise old arithmetic; corrected generation before full replay, then9PASS. One evidence SCP255 retry. No failed T055-V full replay; original T055-A failure/PARTIAL report preserved. Evidence pack21677bytes SHAb0ffa400502fa5b677b120983a73108cb81e5e9f3cbab1de8c943a3b73ac503d verified server/F and local. **REFERENCE_ORACLE_ONLY; zero deployable changes; zero official-test access.** Stop awaiting review; no T056/selfmerge/history cleanup.

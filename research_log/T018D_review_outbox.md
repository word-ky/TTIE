

---

## T018-D review repair — DONE — 2026-09-13T04:57:39.165551+00:00

PR28 comment3998734227 exposed reliance on historical Git objects absent from clean checkouts. Reproduced and fixed: source and archival input bytes now come from current committed files and must match the original pinned receipt hashes and manifest commit/path metadata. No fallback, byte normalization or weaker hash check.

Original PR28 merged as9e3a2447 during repair. The unmerged follow-up is [PR29](https://github.com/word-ky/TTIE/pull/29), branch codex/T018D-verifier-repro, evidence bcb09a05e3c04c44b2b492a54d689b7583674d4d. Only functional edit: research_log/T018D_verify.py. A fresh depth1 checkout391fb87e contains neither historical source6c6eda3a nor scoring4062e01c and exits0. All9 source/6 input/8 artifact hashes pass; independent target-free replay exactly reproduces120 logits/classes/decisions. Evidence: research_log/T018D_review_fix.md, T018D_review_squash.json/log and T018D_review_verification.json.

Selector receipt remains db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94; weights, normalization, scientific source and predictions unchanged. No retraining or GPU experiment. Windows clone setup initially converted line endings; exact committed input retrieval preserves payload bytes, while source/runtime checks remain strict. Original failure logs retained.

New T018-E on main0aa6d639 has been read. After syncing this repair, proceed with separately scoped one-shot fresh qualification using the frozen selector and accepted GPU feature pipeline; no refit or second cohort.

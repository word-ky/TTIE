# T058-AC DONE — first16 full gradients numerically credible

**T058 first16 reverse detail gradients numerically credible under exact clamp-aware chain rule**

All16/16 fixed canonical states passed the unchanged T058-AB gates in order0–15, once. This is numerical credibility of the fixed sanity cohort only. It neither establishes all7,346-state source alignment nor authorizes Stage B; T058-A remains PARTIAL pending research-lead review.

Authorization `ad3c02f202b9fdc35b730c7707bdb2ef75c93b9a`; tested source `35f11ca298dbb5a0dc2104949258657c8c622830`; branch `codex/T058AC-comparison-copy`; PR https://github.com/word-ky/TTIE/pull/92. T058AC_minimal.diff shows the only computational correction: vector_checks uses detached CPUfloat64 copies of g32/g64/g_chain before comparisons. Device/dtype metadata was added to the receipt. All original tensors and the scientific GPUfloat32/CPUfloat64 computation paths remain unchanged. The runner only imports the corrected verifier and binds the preceding T058-AB failure as provenance. No tolerance, graph, renderer, feature cast, scorer/head/checkpoint, direction or cohort change.

The actual mixed-device regression used CUDA:0 float32 g32 (physicalGPU1), CPUfloat64 g64/g_chain, for nonzero and zero cases. Both cases reproduced the old RuntimeError and passed the corrected implementation. Before/after original tensor hashes, devices, dtypes, requires_grad and empty .grad fields matched. Comparison copies were CPUfloat64 with requires_grad false. Exact regression receipt is T058AC_cuda_regression.json. CUDA availability was required; no CPU-only substitute was used.

## Fixed-first16 results

Max full64-D float64 chain error is **0 for every row**. Max GPUfloat32→CPUfloat64 L2 difference is **2.8517847234210123e-6** (row14); lowest nonzero-gradient cosine is **.9999999999452329** (row14), above .9999. Row0 has exactly zero g32/g64/g_chain and passes the original negligible-gradient rule; cosine is null, not artificially assigned1. All image identities and cast-back identities pass, scalar clamp multipliers are deterministically [1,1], energies/gradients finite, inactive tangent and inactive-chain contributions exactly zero.

| index | norm g64 | chain max-abs error | g32–g64 L2 error | cosine | result |
|---|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | N/A (zero) | PASS |
| 1 | 0.267007007027 | 0 | 2.41722590637e-06 | 0.9999999999664325 | PASS |
| 2 | 0.25917479695 | 0 | 2.10763115081e-06 | 0.9999999999685601 | PASS |
| 3 | 0.27517171909 | 0 | 2.05383263159e-06 | 0.9999999999839589 | PASS |
| 4 | 0.278177399758 | 0 | 2.23544885494e-06 | 0.9999999999683064 | PASS |
| 5 | 0.275145242067 | 0 | 1.28579450435e-06 | 0.9999999999891586 | PASS |
| 6 | 0.266635730547 | 0 | 1.73867472464e-06 | 0.9999999999790922 | PASS |
| 7 | 0.275030116245 | 0 | 1.97235851099e-06 | 0.9999999999809347 | PASS |
| 8 | 0.27517171909 | 0 | 2.10885352686e-06 | 0.9999999999823835 | PASS |
| 9 | 0.278719413618 | 0 | 1.23333497103e-06 | 0.9999999999916432 | PASS |
| 10 | 0.292170700461 | 0 | 2.31457811588e-06 | 0.9999999999779586 | PASS |
| 11 | 0.249170203388 | 0 | 1.64596304637e-06 | 0.9999999999792307 | PASS |
| 12 | 0.32318893445 | 0 | 2.37875607054e-06 | 0.9999999999787871 | PASS |
| 13 | 0.296596112582 | 0 | 1.83508808597e-06 | 0.9999999999808635 | PASS |
| 14 | 0.272118832694 | 0 | 2.85178472342e-06 | 0.9999999999452329 | PASS |
| 15 | 0.187205645312 | 0 | 1.75819575269e-06 | 0.9999999999560057 | PASS |

All full original g32/g64/g_chain values, dtypes, shapes, SHA256 hashes and norms, E32/E64, comparison-copy device/dtype metadata, every criterion/error/tolerance/margin and exact boundary counts are stored in T058AC_rows.json. T058AC_summary.json provides compact numeric margins. The original g32 tensors remain CUDAfloat32, g64/g_chain CPUfloat64; only detached comparison inputs are colocated. Zero chain error reflects agreement of the independently invoked frozen-q renderer chain with the full shadow reverse path; this is not an independent AD implementation.

The fixed16 selection SHA is `4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2`; ordering/duplicates/identity were retained. These are canonical source **states**, not16 independent images. Row0 has475200 inactive RGB entries. Active rows have upper-boundary count0; most have interior472151/lower3049 (positive1387/negative1655/zero7), while row14 has interior472150/lower3050 (positive1388/negative1655/zero7). Exact masks and original direction were reused.

## Execution and preservation

Local6tests passed7.73s, compilePASS; server **7passed,1warning in1.75s**, including genuine CUDA regression. One local syntax error (missing brace) was caught during predeployment test collection and corrected before the tested source commit. It did not run an experiment. Server warning was the already-observed NVML initialization warning; CUDA tensor tests and scientific GPU operations succeeded.

Sole release `20260917-105757-ttie-t058ac-copy`, run `20260917-105829-ttie-t058ac-copy`, start2026-09-17T10:58:33+08:00, finish11:00:17+08:00; verifier99.1208476490574s, exit0. Regression passed before the runner started. Exact environment/command is in T058AC_run.sh: physicalGPU1 scientific path, unchanged CPUfloat64 shadow; OMP/MKL/OpenBLAS1 and original frozen arguments. No numerical rerun.

All182source bindings,8T058A/7T058ABhistorical files and before/after scorer/head/checkpoint/legacy/detail/shadow tensors match. Source bank/checkpoint/prototypes/canonical inputs remain frozen. Zero optimizer updates, persistent scientific-state changes, source-clean/JPG access, source-reference gradients/StageB, target-domain and official-test access. Zero FD evaluations and no states beyond16. Prior T058-AB failure receipts were preserved.

A post-run archive SSH connection closed once; retry of archive/transfer succeeded without rerunning science. Due D-disk capacity, the exact committed worktree was deployed directly; raw receipt/logs are retained in the immutable compressed evidence archive and server project, with readable rows/regression/run script also committed locally. Project-root mirrors use hardlinks to avoid duplicate disk blocks.

Evidence archive {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t058ac/T058AC_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t058ac/T058AC_evidence.tar.gz", "bytes": 105798, "sha256": "742341b6cdb18b6956a438a8b96ee9df7d2b2a94c0cf17e27c7ba770fbe5c513", "source182_unchanged": true, "historical8_unchanged": true, "prior_verifier7_unchanged": true, "files": {"train.log": "d0fae872c472ae986320bee038ead2ece229ded317c9e449d5efec8982957d33", "meta.json": "4471d4dbecf88db711f5de8d491dada449b024e1376eb369a5709c349d4fbb1f", "run.sh": "fef4cd2a8b984bfdbb7d722f46c132e3481a60524a307b3aa834e12f98af359d", "artifacts/T058AC_cuda_regression.json": "852c7079097e4b83bbe9698058b88b12298293074dacb52e0ab5704602ef3137", "artifacts/T058AC/receipt.json": "64430933a96208a12073359a38d93bff48d470e5de8d8ece817f9ec570410918", "artifacts/T058AC/selection.json": "4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2", "artifacts/T058AC/states.json": "936b05cfb1660d90deff27b9385dc4aac3cc995d66dbe71b2a762505ab1f3c9b"}}

Home/F/local archive and all seven constituents hash-verified. Full raw hash receipt is inside T058AC_evidence.tar.gz; scientific evidence remains untouched. Recommendation: review the successful first16 gradient credibility result, then issue a separately scoped next task if the full source audit is desired. Stop here; no full-audit continuation, Stage B, readiness promotion or selfmerge.

Both prescribed pairwise probes fail the unchanged five-clause qualification (1/5 each, offset only). The implementation changes only the training objective from pointwise value regression to unweighted within-episode pairwise logistic loss, on the same 40 development IDs, exact five grouped folds and saved 28/30-D candidate features.

| Probe | Spatial MSE | / Region2 | / hard oracle | / matching T016-C probe | Clauses |
|---|---:|---:|---:|---:|---:|
| rank28 | 0.0354018605974 | 1.01022403856 | 1.08716171342 | 0.988694038394 | 1/5 |
| rank30 | 0.0342181611457 | 0.976446106541 | 1.05081128713 | 0.965223273629 | 1/5 |

rank30 improves ranking correlation (median Spearman0.75), but its 2.36% improvement over Region2 misses the required3%; left/right and quadrants also exceed their1% degradation limits. This bounded diagnostic does not establish safe boundary ranking with the specified small scalar-head family. No tuning or follow-on experiment.

Frozen source c91495225b6df73c814ee6f48b7bd3ab8ff2b6c7; final evidence aa71d268294e35f5df67c76eada29f9bec117abe. All ten heads, histories, fold receipts, two frozen OOF tables, source/input hashes and joined evaluation are committed. Four reused C donor files differ only in Windows checkout line endings, documented and bound by the D source hash.

Validation: baseline6testsPASS13.296s; kernel4PASS7.780s; final6focusedPASS12.673s; py_compilePASS. Independent audit verifies63sourcefiles,10inputartifacts,10training-only normalizations,1,000epoch pair permutations,2,160exact saved-head OOF scores and all selection metrics/clauses. Separate SciPy audit verifies all240correlations including9/7nulls. Formal run exit0; no scientific failures or retraining. No GPU, rendering, CLIP, fresh IDs or accepted-reference recomputation.

Full report: https://github.com/word-ky/TTIE/blob/aa71d268294e35f5df67c76eada29f9bec117abe/research_log/T016D_analysis.md

Research review only. No self-merge or PR18 topology repair.
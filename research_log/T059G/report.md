# T059-G — DONE

`T059-E failure is consistent with an inner-held feature-support shift under fixed 1-NN`.

|Query|Rows|Relative Huber (<=0.07650849781930447)|Detail positive (>=0.75)|Detail median cosine (>=0.50)|
|---|---:|---:|---:|---:|
|Inner-train leave-one-image-out|4357|0.06445551663637161 PASS|0.946449339389801 PASS|0.5974024534225464 PASS|
|Inner-held|1529|0.23172274231910706 FAIL|0.9132450222969055 PASS|0.528417706489563 PASS|

This is the fixed second classification: training cross-image support passes all gates and held-out scalar transfer fails. It is compatibility evidence under this one 1-NN diagnostic, not causal proof that feature redesign will fix the problem. No model improvement, benchmark promotion or deployable inference method is claimed.

Authorization 6504bd08f546669e6d6d7558ef59315ac760fbb3. Source on codex/T059G-feature-nn, prefix0f3d5ffa. Sole run20260918-050302-ttie-t059g-nn, 2026-09-17 21:03:06–21:03:18 UTC, exit0. Physical GPU1 NVIDIA RTX A6000 performed all neighbor searches. CPU independently replayed all5886 neighbor IDs, tie counts and distances. Maximum CPU/GPU squared-distance discrepancy7.105427357601002e-15(train),3.552713678800501e-15(held); all indices/tie counts exact. Local3tests3.36s/server3tests1.38s PASS; independent verifierPASS.

Maps were persisted/fsynced at21:03:10.348210UTC, SHA256 a40c269baed073499defcfb651600cc7ff852cc47134cc2cd4e5c2283bb807c4. Train source supervision first opened21:03:11.989057UTC; held supervision21:03:12.046288UTC in a separate evaluation process. Neighbor search used only cached features and image IDs for the required same-image exclusion. No supervision-dependent neighbor choice occurred.

Split remains48train/16held/16excludedouter images and4357/1529/1460rows. No outer supervision opened. Exact Efeature hashes and checkpoint e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0 verified. Frozen x_mean/x_scale also match a recomputation from inner-train features; no model forward. Standardization uses original float32; direct squared differences accumulate in float64 on GPU, with no matrix multiply, TF32, learned weighting or metric tuning. Ascending canonical training-row order plus first minimum gives the smallest global ID for every exact tie, independently verified with NumPy on CPU.

Scalar prediction is the selected training row's saved E delta_t, without recentering the transferred prediction. Detail prediction is the selected training row's existing 64-D reference gradient. Query eligibility is the E double-precision norm>1e-12 rule. Noneligible neighbor directions are zero predictions, never removed:32eligible train queries and4eligible held queries have these neighbors. Eligible query counts4295/1510, ineligible62/19. All scalar rows remain included. Reference vectors were read only from exact previously accepted selected byte ranges; no full mixed cache reads that would expose outer supervision, no Jacobian reads or generation.

Nearest squared-distance train mean/median/p90:6.1469461068418045/4.988656294532646/11.627280033882418; held6.62581756962515/5.8818987193196755/12.502348509808007. Full distance distributions and all240train/80held per-bank plus48train/16held per-image metric records/distributions are in result.json; no subgroup rescue or outlier exclusion.

Source and all opened checkpoint/metadata/feature/row files preserved before/after; selected gradient bytes and metadata match E receipts and were rehashed after. All training/optimizer/model-forward/newimage/newfeature/newreference/outer/target/LOL-v2/official-test/inference-reference counters0. Historical E/F evidence unchanged. No rerun, ksearch, second split, seed, tuning or real-domain TTT. Known NVML initialization warning did not prevent verified A6000 execution; no driver changes.

Commands: python -m pytest research_log/T059G/test_core.py -q -p no:cacheprovider --import-mode=importlib; python -m research_log.T059G.run maps --out <run>/artifacts/T059G; python -m research_log.T059G.run evaluate --out <run>/artifacts/T059G; python research_log/T059G/verify.py <run>/artifacts/T059G. Stop for research-lead review. Large evidence is published from verified memory export because local D capacity is insufficient; home/F and GitHub are authoritative.

# T059-E — bank-relative recipe not supported on fixed nested split

The single fixed nested-development fit completed normally. Inner-held-out relative value Huber and detail median cosine fail; the other three fixed gates pass. No outer C2 supervision was read and no rescue, second split or second training run followed. This is an exploratory source-development result, not fresh benchmark evidence or deployable validation.

| Metric | Inner train (diagnostic) | Inner held-out | Threshold | Margin | Pass |
|---|---:|---:|---:|---:|---|
| relative_value | 0.056902974843978882 | 0.22107574343681335 | <= 0.076508497819304466 | -0.14456724561750889 | False |
| detail_positive | 0.99161815643310547 | 0.868874192237854 | >= 0.75 | 0.118874192237854 | True |
| detail_cosine | 0.65222388505935669 | 0.49155342578887939 | >= 0.5 | -0.0084465742111206055 | False |
| legacy_positive | 0.99767225980758667 | 0.95701056718826294 | >= 0.94999999999999996 | 0.0070105671882629839 | True |
| legacy_cosine | 0.96909910440444946 | 0.90576434135437012 | >= 0.90000000000000002 | 0.005764341354370095 | True |

Absolute Huber is diagnostic only: inner-train0.25344452261924744, inner-heldout0.3741087317466736. Inner-heldout Spearman63defined/17undefined singleton banks, median0.9356521739130435; regretmedian0,mean0.2402098773047328,max6.364080429077148 standardized target units. Full per-bank values and distributions for240training/80heldout banks are in the statistics artifacts. These groups differ from C2, so this result is not a same-cohort performance comparison with C2.

## Fixed split and supervised-data boundary

Immutable manifestSHA `92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125`; first-occurrence80-image grouping, fivebanks/image. Outer groups%5==0:16images/80banks/1,460rows completely excluded. Inner heldout%5==1:16images/80banks/1,529rows. Inner train%5in{2,3,4}:48images/240banks/4,357rows. Zero image overlap; all5,886nonouter rows partition exactly once. IDs, bankindices, canonicalrows and anchors are retained. Training legacy eligible/ineligible4,296/61 and detail4,295/62; heldout legacy1,512/17 and detail1,510/19.

Training opens only inner-training bank files and selected numeric rows from mixed J/reference chunks. Inner-heldout bank files and numeric rows open only in a separate evaluator after final checkpoint persistence. No outer target/reference file or numeric storage range is read, including by hash checks. The reader loads serialization metadata and directly seeks to authorized numeric byte ranges; tests trace actual reads. Source dependencies bind only code and hash-only cache manifests, not prior outcome documents.

Binding detail: full mixed-chunk SHA256 identities are inherited from accepted immutable cache manifests and explicitly labeled `full_chunk_hash_recomputed=false`. Recomputing full-file SHA would read excluded outer-reference bytes, so this task instead freshly hashes only permitted row bytes and serialization metadata before/after, with exact offsets/lengths/globalindices recorded. Selected bank file hashes, input tensor hashes, row hashes, code hashes and before/after checks are all retained. No claim is made that entire mixed files were freshly rehashed in this cycle.

Checkpoint persisted/fsynced/hashed at `2026-09-17T18:36:25.279611+00:00`; separate inner-heldout loading begins `2026-09-17T18:36:35.212525+00:00`. The verifier checks access sets against both allowed and excluded sets, exact persisted ordering, selected byte hashes, numeric gate booleans, checkpoint and evidence hashes. Both outer_supervision_reads and inner_heldout_supervision_reads_before_checkpoint are0.

## Single scientific change and fixed execution

`research_log/T059E/value_loss.diff` shows the original train_head loop with only the value term changed to Huber(p_i-p_anchor,t_i-t_anchor). Anchors are unique state_index0 rows; anchor predictions remain differentiable, including gradients through that branch; anchor rows contribute zero residual. There is no absolute mixture or per-bank parameter. The wrapper reuses unchanged T059B dual_terms for legacy/detail masking and math. Original28-D EnergyHead, fresh seed7 CPU initialization, inner-train-only absolute log-MSE normalization, AdamW, batch256/order,100epochs,finalepoch and1:1:1weights remain fixed. History key train_huber now records the relative value term; diagnostic absolute Huber is reported separately.

Authorization `435549d44e35226afdbb26bcbccea97d48ca3ce5`; exact source `830e80ba0e1a4d09bce9c9ffd57be162a781d013`,179source/Gitblob bindings. Sole run `20260918-023558-ttie-t059e-relative`, release `20260918-023553-ttie-t059e-relative`, exit0. Exactly1source training run/1,800optimizer steps. Local4tests7.79s/server4tests2.86s PASS; local tests include an8-row synthetic training smoke test, not another source experiment. Training stage26.41058858600445s, evaluation5.909828371950425s; separate verifierPASS. No execution failure or scientific rerun.

Commands: fixed one-thread CPU pytest `research_log/T059E/test_core.py`, then separate processes `research_log/T059E/run.py train`, `run.py evaluate`, and `verify.py`, each with `--out <artifacts/T059E>`.

Checkpoint SHA256 `e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0`;100epochhistory, initial/final head hashes, train-only normalization provenance, train/heldout row tensors and full statistics retained. Newimage/newfeature/newJacobian/newreference/target/LOL-v2/test/inference-reference accesses allzero. No historical B/C2/D evidence or PROJECT_STATE changed.

Raw archive 2172975bytes SHA256 `09115e7dfb21eb2d1aff1117943bce1e62155e428cf75d63e0aa236ebd7ca05a`, verified home/F. Compact export bytes were streamed into local memory and every decompressed artifact hash checked before GitHub API publication. Due ongoing Ddisk exhaustion, no large local artifact copy is claimed. Exact server/GitHub copies are authoritative.

Stop for research-lead interpretation. Failure on this one nested split does not prove impossibility for every recipe, but it provides no authorization for outer-holdout evaluation, tuning or real-domain rollout.

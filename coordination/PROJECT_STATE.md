# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement state is insufficient when degradation varies spatially. A compact spatial field can help, but only if four pieces align: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose **gradient field** generates restoration-useful states, (iii) a safe stopping/selection rule, and (iv) a spatial action basis whose support does not unnecessarily couple regions. T012 ruled out selection-only repair of the old semantic trajectory; T013 now shows that source scalar-value regression alone also does not guarantee a useful learned gradient field.

## Current method abstraction

For a test image `x_t`:

1. extract four fixed local quadrant views;
2. obtain the frozen T006 source-trained exposure readout in CLIP feature space;
3. apply the frozen T007 joint clean-abstention gate to determine active quadrants and dark/bright winner;
4. optimize a compact projected EV+gamma state, with inactive regions exact identity and dark/bright EV direction constrained by the frozen gate;
5. evaluate a source-trained differentiable scalar restoration energy from the fixed T013 28-value test-time feature vector;
6. in T014, train that same energy not only to fit source restoration values but also to align `∇_phi E` with source clean-reference restoration gradients;
7. at deployment freeze the energy and backpropagate only through current pixels / frozen CLIP evidence / ISP state; no test label or clean target is available;
8. compare against identity, global/bilinear learned-energy TTT, direct correction, matched projected discrete search, old semantic TTT, frozen source step 16, and a same-data value-only energy control;
9. attach clean-reference metrics/oracles only after all label-free trajectories, selections and outputs are finalized and persisted.

T014 therefore isolates **derivative supervision** while freezing the representation, architecture, action geometry and test-time optimizer used in T013.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** spatial gain is conditional: useful for compatible low-frequency heterogeneous shifts, negligible on homogeneous shifts, and potentially harmful for high-frequency/content-confounded cases. A 96-latent uniform control rules out a trivial parameter-count explanation.
- **T003:** the absolute patch-mean-to-0.5 prior is not identity-safe; identity anchoring + plain TV do not rescue it without destroying useful correction.
- **T004:** zero-shot absolute frozen-CLIP exposure scores show useful dark sensitivity but weak bright discrimination and large content offsets.
- **T005:** a fixed same-image ±0.25 EV relative CLIP probe does not solve the content problem.
- **T006:** source-trained three-prototype readout on frozen CLIP features yields strong exposure ranking/localization, but its first independent gate has excessive clean false activation. These vectors are a discriminative frozen-CLIP readout, not preserved text semantics.
- **T007:** a predeclared joint clean-abstention envelope reduces clean all-view FPR to **6%** and clean image-any activation to **20%**, while retaining usable dark/bright recall.
- **T008:** the first frozen-signal EV+gamma pilot proves the signal can drive real correction, but the original semantic TTT is not qualified: it improves dark/bright and heterogeneous restoration yet fails clean safety and loses to a simple local direct action.
- **T009:** development-only geometry audit shows the semantic gradient is usually directionally useful at identity (94.77% positive alignment; median cosine 0.7973), while stopping/over-correction is a dominant failure. 92.81% improve after the first update and 69.28% have an earlier reference-MSE optimum than the final semantic stop. Gamma removal is not justified. Exact-quadrant `region2` removes major bilinear coupling, but the geometry is synthetic-boundary matched.
- **T010:** residual-evidence target relaxation is a controlled negative result. Zero of 16 fixed rho pairs qualifies: clean safety improves, but heterogeneous advantage over `region2_direct` largely disappears. The original zero-envelope objective retains stronger restoration pressure but is unsafe in the clean tail.
- **T011:** gate-consistent projected action geometry is a controlled fresh-split negative qualification. `region2_ttt_projected` passes 8/10 clauses and reaches heterogeneous MSE **0.02559107**, improving **55.03%** over identity, **46.00%** over projected global TTT and **12.64%** over direct correction, but only **4.46%** over matched projected discrete and misses clean p95 safety (**0.005844413 > 0.005**). Projection changes **67.19%** of updates and **52.72%** of active final coordinates lie on a trust-box boundary. Full TTT beats one-step on 79/80 heterogeneous episodes. The 40%-boundary stress makes hard `region2` **1.30% worse** than projected bilinear, limiting any general renderer claim.
- **T012:** source-trained checkpoint-quality selection is a controlled development-only negative. Stage A passes 1/5 clauses. More importantly, the reference-only oracle over the exact frozen T011 checkpoints reaches heterogeneous MSE **0.02744631**, versus **0.02782951** for source-fixed step 16 and **0.02886186** for projected discrete; the oracle cannot meet either required 5% margin. Selection-only repair is therefore closed for that trajectory.
- **T013:** source-trained scalar restoration energy is a **controlled development-only negative result**, PR #13 squash-merged as `80bafdad5758f62f29dd3257fac3876c37bfda7f`. Stage A passes **3/7** clauses and no fresh Stage B is opened. The learned energy is safe on clean/homogeneous cases (`clean p95=0.00396801`, dark/identity `0.57285`, bright/identity `0.58440`) but its raw-state gradient is insufficiently aligned with restoration: only **54/78 = 69.23%** active non-clean identity gradients have positive cosine and median cosine is **0.24765**. Heterogeneous MSE is **1.21313× projected discrete** and **1.24061× frozen semantic step 16**. Critically, the reference-only oracle over the *new energy trajectory itself* has heterogeneous MSE **0.02896122** and is still **13.96% worse than discrete, 16.54% worse than fixed16, and 18.07% worse than the old semantic final trajectory**. Thus the T013 failure is not primarily a checkpoint selector problem: this value-trained energy generates inferior reachable states. The result does not prove learned energies impossible; it specifically shows that scalar value fitting with the frozen T013 recipe does not sufficiently constrain the derivative field.

## Non-negotiable design principles

- Test-time adaptation, learned-objective optimization and checkpoint selection must never consume test labels, clean targets, degradation masks, gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except in explicitly source/development-side training/diagnostic modules whose learned outputs are frozen before later evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer restoration or task utility from lower self-supervised/learned energy alone.
- Do not infer useful test-time gradients from scalar source value-regression accuracy alone; derivative quality must be checked explicitly when the learned scalar is optimized through the input/state.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate representation quality, gate safety, inner-objective gradient alignment, stopping/selection, action geometry, spatial basis and optimization mechanism before adding larger fast models/meta-learning.
- After a result motivates a new hypothesis, use a fresh deterministic source/calibration or evaluation split; inspected images are unavailable for corrective tuning.
- Source-trained controllers/objectives/signals may use clean references only during source training; source labels, clean targets, Jacobians or synthetic degradation metadata must never become test-time inputs.
- A gradient-based TTT claim must beat non-gradient direct/discrete actions using the same frozen signal. If simple policies match or win, do not attribute value to TTT.
- A stopping/selector claim must be checked against a reference-only trajectory oracle. If the oracle itself cannot clear the target baseline margin, do not keep tuning the selector on that trajectory.
- A learned-energy claim must likewise inspect the reference-only oracle over the learned trajectory. If the oracle is weak, repair trajectory generation rather than selection.
- Hard region-aligned fields remain explicitly limited by boundary-misaligned stress evidence until a more general spatial basis is tested.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

## Milestone M2 — simple regularization of absolute prior

Status: **COMPLETED / T003 NEGATIVE DIAGNOSTIC**

## Milestone M3 — degradation-aware semantic readout and first correction pilot

Status: **COMPLETED THROUGH T008 — READOUT QUALIFIED, ORIGINAL TTT NOT QUALIFIED**

## Milestone M4 — objective/action geometry diagnosis

Status: **COMPLETED / T009 DEVELOPMENT-ONLY DIAGNOSTIC**

## Milestone M5 — residual-target stopping repair

Status: **COMPLETED / T010 NEGATIVE DEVELOPMENT GATE**

## Milestone M6 — gate-consistent projected action geometry

Status: **COMPLETED / T011 CONTROLLED NEGATIVE QUALIFICATION**

T011 establishes a strong but still unqualified projected spatial trajectory and shows that projection is genuinely active. It does not justify further hand-tuning of the same action box or a general hard-quadrant renderer claim.

## Milestone M7 — source-trained label-free trajectory stopping

Status: **COMPLETED / T012 NEGATIVE DEVELOPMENT GATE**

T012 shows that even perfect selection over the frozen T011 checkpoints cannot provide the required margin over fixed16/discrete. The next mechanism had to change the trajectory itself.

## Milestone M8 — scalar source-trained task-aligned inner objective

Status: **COMPLETED / T013 NEGATIVE DEVELOPMENT GATE**

T013 is the first learned-inner-objective experiment. The scalar energy fits source restoration values and remains identity-safe enough, but it fails explicit source-calibration gradient-alignment and strong-control utility gates. Its trajectory oracle is itself worse than discrete/fixed16/old semantic final. This closes further tuning of the same value-only energy recipe on inspected data.

The 80 T013 train + 20 calibration images are permanent development data; total inspected IDs through T013 are **508**. No T013 fresh Stage B split exists.

## Milestone M9 — derivative-supervised learned inner objective

Status: **ACTIVE / T014 OPEN**

T014 holds the T013 28-feature representation, SiLU MLP architecture, state bank, T011 action geometry and test-time 40-update optimizer fixed, and changes only source training supervision. A Sobolev-style energy is trained with both scalar restoration-value loss and explicit source clean-reference **raw ISP gradient-direction matching**. A same-source value-only head is trained as a causal control.

T014 uses a fresh 80-train/20-calibration source split excluding all 508 previously inspected IDs. Stage A must demonstrate both restoration-gradient alignment and actual heterogeneous advantage over matched projected discrete, frozen semantic step16, and the value-only control before any new fresh evaluation split is permitted.

No feature expansion, learned spatial basis, detector work, meta-initialization, prompt retraining or ViT3-style fast model is authorized until this derivative-supervision hypothesis is resolved.

## Open task

`T014 — Source-Supervised Sobolev Restoration Energy` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- Feature-sufficiency audit / richer test-time representation only if T014 shows derivative supervision still cannot align gradients while the frozen 28-feature schema is held fixed.
- Soft/learned or low-rank spatial basis after the T011 boundary-misaligned stress result is addressed explicitly.
- Jointly learned fast initialization only after a learned inner objective demonstrates value over direct/discrete controls.
- Frozen detector/object-feature consistency only after a corrected restoration mechanism qualifies.
- True CLIP-LIT/CoOp-style prompt learning or a non-CLIP degradation encoder if the current frozen-feature readout becomes limiting again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.

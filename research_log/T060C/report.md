# T060-C — BLOCKED: requested initialization/gain-only swap does not uniquely match accepted T036

Authorization: main `31a7bd3686ad8f9c2729611c4a48ecd7508fe241`. Accepted T036 source: `f80cea4c9d8186e0c4a0404b28ccd58c5e1b5678`. This is a procedure-contract finding before implementation or real-data inference; it is not a negative finite-step result.

The task explicitly requires preserving the accepted T036 initialization, update and stopping semantics, reusing “accepted T026-A starting outputs/states,” and substituting `J_gain^T q_E` into common-gain updates. The accepted executable procedure differs from the assumed sequential gain-only setup:

1. `scripts/run_t036a.py:44–49` passes the same **raw low image** separately to `gamma_range_ttt.trajectory` and `common_gain_ttt.trajectory`. The common path never loads the baseline output or selected T026 state.
2. `ttie/common_gain_ttt.py:23–27` creates a fresh CommonRegion2 and Adam over **the full model.raw**, lr0.03. `ttie/common_gain.py:11` initializes raw to zeros of shape1×3×2×2. The procedure jointly adapts4 EV,4 gamma and4 gain coordinates from identity for40 updates (or zero if no active region).
3. `ttie/common_gain_ttt.py:41–46` differentiates one scalar head energy with respect to all12 raw coordinates, assigns the full gradient, then applies Adam and CommonBox.
4. `ttie/common_gain_ttt.py:56–58` also uses that scalar head energy to choose the minimum-energy state, earliest tie. Thus the head supplies both the full gradient and state selection, not only a gain gradient.

Feeding only the four-coordinate `J_gain^T q_E` into this twelve-coordinate procedure requires an additional scientific choice: retain original T014 EV/gamma gradients, replace all coordinate gradients with T059-E, or freeze EV/gamma at selected T026 states. The first is mixed guidance (the task forbids blending); the second is a full twelve-coordinate head swap; the third changes initialization and the accepted joint trajectory. Scalar energy selection also needs an explicit corresponding head choice. These are different experiments and cannot all be described as preserving the same procedure.

The most direct interpretation of “replace only the head, preserve exact T036” would be: keep the original raw-low/identity initialization, joint12-coordinate Adam0.03×40, CommonBox and minimum-energy/earliest-tie semantics; substitute T059-E consistently for both full-state gradient and energy-based selection; explicitly compute a full28×12 Jacobian and record its gain slice. **This is a proposal for clarification, not an executed method change.**

Following the task's explicit stop-rather-than-guess instruction, no100-image T060-C inference, normal/reference opening, per-image outcome-driven selection, target fitting or official-test run was started. Only pinned-code inspection and a deterministic synthetic baseline contract reproduction were performed. `source_binding.json` gives exact Git blob/SHA identities; literal accepted sources are included as `.txt`; `contract_evidence.json` records the synthetic reproduction. Prior T060-B remains accepted and unchanged.

Requested next action: research lead reissue/clarify T060-C initialization, guidance for EV/gamma, and energy-based state selection. If full-head replacement is intended, explicitly authorize the original joint12-coordinate path above; otherwise specify the changed gain-only procedure and its comparison protocol. No user permission for repository work is needed; this is a missing scientific definition in the coordination task.

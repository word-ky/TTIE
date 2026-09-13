# T019-C final deadband selector frozen — engineering checks passed

Exactly one final x head and one final y head were trained once on all120 accepted development feature rows. The six requested engineering checks pass. This is an **engineering freeze only**, not a performance experiment, fresh qualification or new generalization evidence. No training-set agreement, selected MSE, family safety or oracle metric was evaluated. No T018-E data were used.

## Frozen artifact

- Source commit: `a258a72bf4d317efe92db8723f2070b7c2120f7f`.
- Single freeze receipt: `research_log/T019C_run/selector_frozen.json`, finalized **2026-09-13T08:33:19.778940Z**.
- Receipt SHA256: **0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77**.
- x head SHA256: `7475d1582bdd68ea3542fdee937e9f6db75e78d55a5b36364262161b6351beac`.
- y head SHA256: `551419ff22c6714d9f2114184c0a29081aec3775ae48c084a0467e6c4b1e03da`.
- Normalization SHA256: `bb011113247239e6850bd3086e62609a9fe68e375234766ca44746cf62b2b305`.
-120-row replay SHA256: **98ee9d17ff5670b2a8bbde2672db93a970bc2bd1e79553ff649106ed0d43c73e**. Saved/reloaded logits, classes and choices are exactly equal.

The single receipt binds source/inference file hashes, all feature and target origins, donor source, runtime, the complete recipe, all120-row normalization, both weights,100-epoch histories, replay features/results and config. `selector_frozen.sha256` pins that receipt for subsequent loading.

## Inputs and unchanged method

The accepted T019-B config is taken from merge `e04a96da31a1a2f359e45ba5f251e04989ab895d`, SHA256 `3a7d3f9949d7d3be84aa558961fafe4e7c7559529bcc3e3340a71bd59f29a2e8`; its training source is `c66a8a0f5f16d64bf160c74e6217a652d0950dc4`. All donor source files listed by that config remain byte-identical. The features are the original T016-B saved scoring artifacts at `4062e01cb93de731c394015c5ac741d6c08e04d8`; selection SHA256 `99dbb10260e5045d5c5deb1d53b5f68d42567a02d5a1b898b234e6b5d113b866`. Complete paths and hashes are in the freeze receipt.

Labels are the unchanged accepted T019-A artifact `research_log/T019A_run/decisions.json` at PR32 head `91f750e871d2c133624bbe4972f3fb3086f25a6c`, SHA256 **d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45**. The existing reader extracts only bx/by for all120 development rows; no label recomputation or nested reference-MSE decoding occurs. This is declared source/development fitting.

Both heads use the literal T019-B/T018-C recipe:84→64→64→3, SiLU, unweighted cross-entropy, AdamW lr1e-3/weight decay1e-4/betas(.9,.999)/eps1e-8, batch256, seed7,100epochs, finalepoch only. Per-axis mean and population std are computed from all120 rows; std is clamped at1e-12. Class order is center/low/high `[.5,.4,.6]`, including the inherited deterministic tie rule. Runtime is Python3.12.7/Torch2.13.0+cpu on the original Windows backend. No GPU requirement for these small fixed heads; no architecture, optimizer, seed or normalization change.

## Reference-free inference API

The existing `ttie.direction_selector` is reused byte-for-byte. It accepts five28-D vectors (one row or batch), in this order:

| Argument | Candidate |
|---|---|
| center | bx=.5,by=.5,tau=0 |
| x_lower | bx=.4,by=.5,tau=0 |
| x_upper | bx=.6,by=.5,tau=0 |
| y_lower | bx=.5,by=.4,tau=0 |
| y_upper | bx=.5,by=.6,tau=0 |

```python
from ttie.direction_selector import load_selector

selector = load_selector(
    'research_log/T019C_run',
    '0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77',
)
rows = selector.predict(center, x_lower, x_upper, y_lower, y_upper)
```

Internally each axis is84-D `concat(f0,fminus-f0,fplus-f0)`. Returned records contain x/y logits, class indices, bx/by and inherited hard-candidate indices. There are no target, reference, family, condition, gain/mask, oracle or image-ID arguments. Loading reads only the pinned receipt, inference source bytes and two checkpoints; it needs no training loader, target artifact or Git access. The API signature and output convention are unchanged from accepted T018-D.

## Verification

| Requested check | Evidence |
|---|---|
| Exactly2 heads,1 fit each | One formal train command; x/y100-epoch histories; focused fit-call/input test; frozen receipt |
| Complete immutable binding | Single pinned receipt includes runtime, source/feature/target/normalization/checkpoint/replay hashes |
| Reference-free signature | Existing five-feature API; focused signature assertion; no reference arguments |
| Metadata mutation/removal invariance | Frozen inference remains exact after changing/removing target, reference-MSE, condition, family, semantic-ID, mask, gain and oracle artifacts; Git/data access forbidden in test |
| Independent exact120-row reconstruction | Standalone PyTorch reconstruction in a five-file directory with no targets/references; no TTIE imports |
| Exactly2 heads and literal recipe | Verifier checks two checkpoint files, bothfit counts, exact full recipe, state reconstruction and all120-row normalizers; no extra model or calibration layer |

Baseline10 tests passed20.58s. Final affected tests: **11 passed14.88s**. Formal training, saved-head replay and independent verification all exited0. API replay completed08:33:25.821988Z; independent verification completed08:33:48.957545Z. The independent verifier opens only four allowlisted feature/donor-config origins, source bytes and frozen engineering artifacts; it never opens the T019-A target artifact or reference metrics. Neural reconstruction opens exactly receipt, two heads, replay features and expected replay. Per-row logits/classes/choices and normalization match exactly120/120.

Commands (D:/anaconda3/python.exe, from project worktree):

```text
python -m pytest tests/test_direction_probe.py tests/test_direction_selector.py tests/test_direction_selector_run.py tests/test_deadband_probe.py tests/test_deadband_selector.py -q
python -m ttie.deadband_selector_run train --output research_log/T019C_run --source-sha a258a72bf4d317efe92db8723f2070b7c2120f7f
python -m ttie.deadband_selector_run replay --output research_log/T019C_run --receipt-sha 0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77
python research_log/T019C_verify.py
```

No scientific or runtime failures, no recipe deviation, extra fit, calibration, fallback, threshold search, rerendering, CLIP/TTT, fresh images/manifest, downstream benchmark or qualification. No T018-E references/features/logits/outcomes used. Stop here for research-lead review; no T019-D or later-stage experiment has begun.

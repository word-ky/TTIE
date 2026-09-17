# T058-AA DONE — row3 clamp backward convention explained

**T058 row3 reverse gradient explained by mixed-boundary clamp backward convention; prior aggregate interval criterion invalid**

All six predeclared criteria pass. This is a numerical-verifier conclusion only: T058-A remains PARTIAL, the 7,346-state audit remains stopped, and source readiness/Stage B are not established or authorized by this result.

Authorization `f53adf3f11ab5033960ad53763a252ef7c74f373`; tested source `e636a4379e74f2d39961da349ea873ca41829d80`; branch `codex/T058AA-clamp-decomposition`; PR https://github.com/word-ky/TTIE/pull/90. Only verifier files under research_log/T058AA* were added. The original T058-A source `aa22caacd42906ba36063a1a2560600ba2370897`, stopped evidence `e7953a202112baf647654411626e743865ae8f25`, and T058-Z shadow/source/evidence were preserved. All 168 deployed source bindings, 8 historical T058-A files and 6 T058-Z run files match before/after/archive.

Only canonical index 3 (bank1/state2/image36660) ran. The donor first16 selection is retained solely for provenance (SHA4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2); no other row was evaluated. The original GPU1 float32 path and nonpersistent CPUfloat64 scorer/head/detail copies were reused unchanged. The final image reconstructed from the observation-only pre-clamp tensor is bitwise identical to the unchanged shadow renderer. The accepted same-input feature cast-back identity also remains true. No cast, backend or scientific graph was changed.

The exact directional tangent is frozen detail times bilinear interpolation of the fixed alternating unit-L2 direction, with inactive tangent zero. The downstream q is the float64 gradient of the unchanged frozen energy with respect to the independent clamped-image leaf. No reference image or reference gradient is involved. Scalar CPUfloat64 clamp backward at 0 and 1 returns [1,1] identically in both microprobe evaluations.

| Contribution / prediction | Value |
|---|---:|
| I | 0.0023723224254001479 |
| B_plus | 0.00070702900673095365 |
| B_minus | 0.0032691154759859373 |
| zero_tangent | 0 |
| d_rev_pred | 0.0063484669081170387 |
| d_plus_pred | 0.0030793514321311014 |
| d_minus_pred | 0.0056414379013860847 |
| surrogate_derivative | 0.0063484669081170422 |
| actual d64_rev | 0.0063484669081168201 |

Thus reverse = I+B_plus+B_minus; positive direction = I+B_plus; negative direction = I+B_minus. Both boundary contributions are positive and nonzero, so reverse exceeds both one-sided first-order predictions. This is the exact chain-rule decomposition of the framework's inclusive boundary backward convention, not an inferred finite-difference explanation.

The actual row has **only lower-boundary pixels**, with opposite tangent signs: interior472151, lower3049, upper0; lower positive1387, lower negative1655, lower zero7. The nonzero boundary count is3042, zero-tangent contribution0; outside/inactive counts0. Therefore mixed directional activation at a single boundary already suffices; the explanation does not require simultaneous lower and upper boundary pixels. Masks use exact comparisons without epsilon.

| Criterion | Error / outcome | Tolerance | Margin | Result |
|---|---:|---:|---:|---|
| Reconstructed image identity | bitwise exact | exact | exact | PASS |
| Scalar clamp convention | [1,1] twice | exact inclusive | exact | PASS |
| reverse_reproduction | 0 | 1.634846690811682e-10 | 1.634846690811682e-10 | PASS |
| reverse_decomposition | 2.1857515797307769e-16 | 1.4696933816233641e-07 | 1.4696933794376125e-07 | PASS |
| clamp_surrogate | 3.4694469519536142e-18 | 1.4696933816233641e-07 | 1.4696933815886696e-07 | PASS |
| Mixed boundary contributions | both nonzero; zero-tangent=0 | exact groups | satisfied | PASS |

E64 and independent-leaf energy both -4.796321043571391. d64_rev exactly reproduces T058-Z .00634846690811682. Original float32 E32=-4.796320915222168 and d32_rev=.006348521463223733; its difference from T058-Z is ~9.36e-9, while CPUfloat64 is exact. No threshold was adjusted.

Frozen T058-Z h=.0005 secants are quoted as historical context only: positive .0030797878665111966 and negative .005641091338048909. They are close to the analytic directional predictions, but no new FD evaluation or ladder was run and no acceptance criterion depends on selecting a step. The prior aggregate interval criterion is invalid as a necessary condition here; the preserved T058-Z negative receipt remains correct under its then-specified contract.

Local4tests passed5.35s; server4tests passed1.52s; compilePASS. Sole release `20260917-082738-ttie-t058aa-clamp`, run `20260917-082811-ttie-t058aa-clamp`, start2026-09-17T08:28:17+08:00, finish08:28:47+08:00, verifier26.4267571719829s, exit0. Exact command/environment is in T058AA_result/run.sh: GPU1 original and CPUfloat64 verifier, same frozen source-bank/checkpoint/prototype/selection/stopped-run arguments as T058-Z.

Before/after scorer/head/original detail/legacy/nonpersistent shadow/source/checkpoint hashes agree. Zero optimizer updates, persistent scientific changes, JPG/clean opens, Stage B, target-domain and official-test access. No FD evaluation, other rows, retraining, alternate backend, tolerance, cast, precision or rerun.

No scientific exception/failure occurred. One postprocessing archive attempt failed because the helper's prior-run path was mistyped as 20260917-074144-ttie-t058aa-cast; the literal path was corrected to the actual T058-Z run and archive/hash verification succeeded. The experiment was not rerun and no source/evidence was modified.

Evidence archive {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t058aa/T058AA_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t058aa/T058AA_evidence.tar.gz", "bytes": 69378, "sha256": "394e8d8a378eff0fbcbabcbe34e5e4df26de80484f4aa6e55588769074de1362", "source168_unchanged": true, "historical8_unchanged": true, "prior_verifier6_unchanged": true, "files": {"train.log": "9adf43a67039868d1e925549b50a00147570d116b490f288f46b02553e71f1b5", "meta.json": "cd0d95ed5c6eed94389d34111d3845d746fafa588ba84a60d28c947081276ac6", "run.sh": "72ce01594fcefca9e961178169a432cbc0f6d1d4dc446ba806af54c86cd00a52", "artifacts/T058AA/receipt.json": "4d788787600da06043a20ec33c9e9f1e5d7fbae77c61419e7739664a90275b7a", "artifacts/T058AA/selection.json": "4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2", "artifacts/T058AA/states.json": "d5316ac5dacfafa5e8606fa7abe82c1d4a03ba719997f2749cd3c61ece7effb2"}} verified home/F/local; all six extracted constituents match hashes.

Recommendation: research lead may now adjudicate this row3 numerical explanation and decide whether a separately scoped continuation is warranted. Stop here; no self-merge or T058-A/Stage-B continuation.

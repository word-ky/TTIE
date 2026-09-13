# PR29 original provenance repair

Review3998783968 is valid: repeating origin labels in an archival manifest is not a resolution of those origins. The verifier now loads a minimal185272-byte pack of34 actual Git commit/tree/blob objects into an isolated bare repository, resolves all nine original scientific source locations and six original input commit:path locations, then requires byte equality with the recorded SHA256s and committed copies. The pack contains the exact recorded commits; it does not rely on parent history, a remote request, or the caller repository containing those otherwise-unreachable objects.

Three focused tests PASS: all15 actual origins match; incorrect commit rejected despite unchanged payload/hash; incorrect path rejected despite unchanged payload/hash. Full verifier PASS with120 exact reference-free logits/classes/decisions, original source/head/receipt bytes unchanged. No training, fresh data, evaluation or GPU work. A shallow clean-checkout test follows before delivery. Pack SHA256 b2077995266ffe7d7914cfc0a511ab25adac6cb74956476e7a92734e3d69ebe9.
2026-09-13T06:00:56.971468+00:00

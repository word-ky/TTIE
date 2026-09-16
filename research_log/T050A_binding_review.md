# T050-A source-binding review correction

The current `T050A_source_binding.json` now binds the corrected replay script at the reviewed branch. All 62 entries were verified against exact committed Git blobs; the only changed entry is replay.py. This repairs the observed preflight/run source-check failure for a new reproduction checkout. No GPU experiment, normal decode, metric evaluation, or parameter update was performed for this repair.

`T050A_tested_source_binding.json` preserves the exact original manifest from tested scientific source9109f3296716a378a8d1424360ccec2c813a9713. Existing T050A_result/preflight and audit/config/freeze remain immutable historical records of that source. Their hashes must not be rewritten to describe a run that never happened. The delivered corrected replay ran separately as023442 against these frozen results, as recorded in the completion report.

The old recovery package remains an immutable receipt of head1160d117; this review patch is an additional Git commit and separate review recovery artifact. New reproductions should use the current manifest; historical audit should use the tested manifest and recorded source SHA.

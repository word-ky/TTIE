# T072-B native UHD-LL preflight

Authorization main ffb42f7f9727833250ea77804bd71eb3b318ce44. This task supersedes stale PROJECT_STATE current-task text through the newly published OPEN inbox. LSRW stays pending.

Pinned author README identifies original UHD-LL with 150 test pairs. Google Drive metadata enumeration uses public file-list metadata only, no gt file downloads. Both complete lists contain 150 distinct matching names. Smoke selection was persisted before downloading pixels: lexicographic first 1003_UHD_LL.JPG, native3840x2160 RGB.

Reuse: exact T070A FinalOurs API and ReadScope; unchanged accepted Retinexformer/SNR exporter main and T071B bindings. Only task wrapper and metadata/output audit are new. No method source/config change. Each method gets its own process, same input, native geometry. Execute Ours then Retinexformer then SNR, once each; stop at first failure, mark later methods unrun. Do not reduce resolution, tile, switch precision, tune memory allocator or modify method to fit. Record actual available memory to distinguish resource failure from intrinsic method incompatibility.

Tests: paired metadata/selection timing and frozen Ours/baseline affected tests, then remote identical tests; full scientific preflight itself is one-shot. Independent audit reopens saved outputs only, no model rerun. An audit PASS with classification BLOCKED verifies the failure receipt, not feasibility acceptance.

Serialization only: clone the selected CPU image before saving to avoid preserving the full trajectory backing storage; pixel values and method state are unchanged.

No references/metrics/training/model fits. A completed preflight is not a full benchmark. Stop after pass or exact failure and report.

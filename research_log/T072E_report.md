# T072-E — UHDLL_FAIR_HARNESS_SEALED

The complete UHD-LL fair-benchmark harness is sealed as two separate stages. The inference/freeze module accepts only the canonical 150 declared low-image rows, rejects reference/gt/clean/label/metric paths, validates exact three-method coverage and native `[3840,2160]` finite outputs, and writes a content-hashed freeze manifest. The evaluation module accepts only a verifier-approved complete manifest and is the only stage with a reference mapping argument.

Focused synthetic tests pass (`10 passed in 0.54s`). They cover missing and duplicate images, altered output hashes, altered bindings, wrong geometry, incomplete method coverage, freeze tampering, attempted `gt` read, and evaluation refusal for incomplete freeze. Canonical real-low enumeration reproduces exactly 150 names and the bound pair-manifest SHA without decoding any reference payload. No real model inference, optimizer, reference read, or metric was performed: `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `real_reference_reads=0`, `real_metrics=0`.

Task files are under `research_log/T072E/`; no scientific method or preprocessing was changed. The next cycle may decide independently whether to retry the clean-GPU baseline smoke or launch the full frozen benchmark.

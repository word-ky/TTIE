# T072-P — NATIVE4K_RUNTIME_ASSETS_VERIFIED

One read-only asset audit on the authorized A6000 host verified70/70 unique sealed paths: every file exists, is regular/readable, has recorded byte size and matches its expected SHA256 exactly. No asset mismatch or repair occurred.

Runtime root is `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines`, the explicit accepted T071-B release path preserved in project T071B_state.json. This is the concrete runtime-root selection for the future T072-O launcher. Absolute checkpoint/source paths retain their accepted /home locations; relative bindings/config/exporters resolve under this release. No symlink or current-release alias is substituted.

Manifest derives only from T072-O evidence4eaa37e7fce5dba59a5bb769363d061cc340bac3: spec SHA4a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895, evidence rootc000543fc2623f76d1270d7c479114ea325ad3971d12821bbf46bcd8629e8c9d. It contains both accepted binding files, every file enumerated under both baseline files maps, plus the original T071-B invocation/provenance source entries. Duplicate exporter paths are deduplicated, preserving their frozen hashes. Expected allowlist canonical SHA256a42adfc826e16f05fe9bb789ed3070e771f18a8821e86690043f59ef0b40bd5c.

Both exporter source bytes, both accepted binding files and configs, and both model checkpoints match. Retinexformer checkpoint6478393bytes SHA539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b; SNR checkpoint156523164bytes SHA432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781. All additional pinned official sources/archive/license/metadata also match; receipt.json retains per-file results.

Python executable `/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python`, version3.12.12. Top-level static find_spec resolves torch,numpy,cv2,PIL,einops; exact origins are recorded. Exporter/model imports and runtime ABI/CUDA functionality were NOT_EXECUTED_BY_DESIGN. These static checks establish asset and package discoverability sufficient for a later attempt, not successful imports/forward execution or native4K feasibility. The sealed T072-O launcher remains a separately supplied artifact, not installed into the historical T071-B runtime by this audit.

Audit read only explicitly enumerated assets via stat/open/hash. It did not read any target low or reference image or initialize/import CUDA/model modules. gpu_queries=0, cuda_initializations=0, inference_runs=0, input_payload_reads=0, reference_reads=0, metrics=0, process_interventions=0. No installation, environment change, source edit, copying or repair on the host.

Collector audit.py and audit.ps1 preserve exact execution: the existing Invoke-AutodlSsh helper invokes the pinned Python executable with a base64-encoded in-memory collector and sealed JSON allowlist; only local receipt output is written. The full70-row raw receipt is retained. Three synthetic tests passed (0.013s): missing file, wrong digest, target/GT/clean path injection rejection before access. Independent verify.py reconstructs expected path/hash union from immutable T072-O Git artifacts and replays all70 receipt rows and counters; PASS. It does not contact the host. Reproduce locally with `python research_log/T072P/test_audit.py` and `python research_log/T072P/verify.py`.

Local verification initially rejected an accepted zero-byte source file because it required size>0. Corrected this to size>=0 while retaining exact SHA checks; no remote repeat or file change was needed.

Authorizationec485a97. PROJECT_STATE.md and all frozen scientific/analysis artifacts unchanged. Stop after reporting; actual smoke requires a later instruction and qualifying GPU.

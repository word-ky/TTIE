# T072-A — BLOCKED

Canonical LSRW evaluation acquisition is blocked before inference. The author's official corrected-name dataset share was resolved and its extraction code accepted, but the canonical `Eval.zip` bytes could not be obtained. Therefore exact complete pairing, per-file hashes and native geometry cannot yet be established. No LSRW low or reference pixels were opened, no method ran, and no metrics exist for this task.

## Verified provenance

- Author repository: https://github.com/JianghaiSCU/R2RNet (the paper's historical abcdef2000/R2RNet URL redirects here), pinned commit `cf59012d276415bddb257e76af66b8f15b5193e2`.
- Pinned README SHA256 `06fbc0d97e057e2e87e5200567d6f0c2d55fe8b11a21923132d98cc34ac57aa2` links the corrected-name LSRW archive at https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA with public extraction code `wmrr`.
- The original paper's Dataset section describes 5,600 training pairs and 50 evaluation pairs: https://arxiv.org/html/2106.14501. This establishes the expected count, not a file-level verified manifest.
- The author share accepted the public code (HTTP200, errno0) and listed `Eval.zip`, **18,464,910 bytes**, and `Training data.zip`, 1,754,809,582 bytes. The training archive is unnecessary and was not downloaded. Listing metadata is in `official_listing.json`. Its `md5` fields are opaque provider strings, not validated MD5 digests; no archive SHA256 is claimed before obtaining bytes.
- The pinned GitHub tree contains only `data/test/low/46.png` and `490.png` under its example data directory, with no paired normal directory. The complete canonical test set cannot be reconstructed from those examples, and they were not substituted for it.

## Observed acquisition failures

The author landing page and extraction-code step succeeded. An ordinary anonymous shared-file download request for the listed Eval.zip returned HTTP200 with **errno2** and no file payload. The returned message was empty; this report does not assume whether the cause is a missing session, request parameter, authentication requirement or another provider restriction. `download_attempt.json` records the observed response without session secrets.

The browser control entry point was also tried twice against the official URL. Both attempts failed before a browser could be controlled: `failed to write kernel assets: system cannot find the specified path (os error 3)`. Thus no interactive authenticated download could be completed through that tool.

Bounded metadata searches found no LSRW path under the configured server project shared directories or the two wjq roots to depth5. Project-root/.autodl immediate acquisition inventory likewise contains no named LSRW archive. A broader local text search produced no useful evidence and was stopped; this is not a claim that no other disk anywhere could contain a copy.

## Scope and validation

Authorization: main `0632d019edaf8b5a8ba7d5e50a910f2c84001b38`, current T072-A inbox. T071-B acceptance is acknowledged. Final Ours remains the T070-A immutable artifact; Retinexformer and SNR-Aware remain the exact accepted T071-B LOL-v2 checkpoints/configurations. No scientific source, binding, geometry convention, preprocessing, metric or hyperparameter was changed. UHD-LL was not accessed.

The task's stop criterion requires BLOCKED when canonical provenance/pairing cannot be established. Author identity and archive identity are located, but archive contents and the complete pair manifest are not yet verified. This is a data-access prerequisite failure, not a negative cross-domain result or a demonstrated model geometry failure.

Checks performed: official repository commit/tree/README retrieval; README hash; successful official extraction-code response; listing metadata; rejected download receipt; existing-data path checks. No experiment implementation was added, so no new unit tests were warranted. **Inference runs0; optimizer runs0; model fits0; LSRW reference reads0.** Independent metric verification is not applicable because no outputs or metrics were generated. Partial samples, unofficial mirrors and paper numbers were not used.

## Concrete recovery

Obtain the author's **Eval.zip only** from the verified official share (public code `wmrr`) and place the unmodified archive at either:

- local `D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement/.autodl/LSRW/Eval.zip`; or
- server `/media/wenchang/F/wjq/TTIE/shared/t072a/Eval.zip`.

Then resume this same bounded task by verifying archive identity, enumerating the complete official 50-pair split and original geometry without reading reference pixels, and implementing the three frozen inference wrappers. Any reference hashes that require payload access must remain in the postfreeze evaluation phase; archive/central-directory metadata can bind the reference list beforehand as in T071-A. Do not silently substitute an arbitrary 50-image subset. No method rerun or target-specific tuning is authorized by this blocked report.

Task-owned evidence is in `research_log/T072A/`; durable state/report/HANDOFF are under the project root research_log. This stacked evidence branch is for reviewing T072-A files, not merging the full historical diff.

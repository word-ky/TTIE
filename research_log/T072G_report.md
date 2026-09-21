# T072-G — BLOCKED: SMID RGB cohort is not one sealed cross-domain protocol

## Result

`BLOCKED` is the required classification. The official SNR-Aware source provides a reproducible ordered list of 49 test scene IDs, but the accepted SNR-Aware and Retinexformer loaders do not define the same low-frame cohort. The A6000 host also has no `SMID_LQ_np` payload from which a low-only manifest, geometry, dtype, coverage count, or per-file SHA256 list can be sealed.

No clean/long-exposure/reference payload was opened or decoded. There was no model inference, optimizer, fitting, metric calculation, or checkpoint change. The machine-readable evidence is in `research_log/T072G/manifest.json`; `verify_manifest.py` checks the root hash and the no-reference-read ledger.

## Provenance

- SNR-Aware is frozen at upstream commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`. Its official `test_list.txt` has 49 ordered four-digit scene IDs, 49 lines, 245 bytes, and SHA256 `cd408769f80576bba9d4057add2d7c99e70bbf62682af4d7207885ab0d631fc3`.
- Retinexformer is frozen at upstream commit `1e9a0efce4b306b6701b824768370ff26066c32a`.
- The public processed-data links are recorded in the manifest. They describe archives containing `SMID_Long_np` and `SMID_LQ_np`; no archive payload was opened during this task.

## Material protocol discrepancies

1. `data/dataset_SMID_test.py` in SNR-Aware sorts each low scene directory and truncates it to `img_paths_LQ[0:30]`. It builds a five-frame `new_info` temporal window and pairs each low index with the first filtered GT path.
2. `basicsr/data/SMID_image_dataset.py` in Retinexformer sorts each low scene directory but does not truncate it. In test phase it reads one low path `[idx:idx + 1]` for every low index, again pairing to the first filtered GT path. Its configuration still declares `N_frames: 5`, so the two accepted baselines do not share one exact ordered frame cohort.
3. SNR-Aware hard-codes `/mnt/proj3/xgxu/EDVR/datasets/test_list.txt`. Retinexformer computes `dirname(dataroot_gt)/test_list.txt`, which is `data/SMID/test_list.txt` under its shipped config. Retinexformer's README separately asks for `text_list.txt` under `data/SMID/SMID_Long_np`. These names and locations are not bound to one canonical file by the public recipe.
4. Both readers load processed `.npy` arrays, divide by 255, and reverse channels with `[2,1,0]`. The SNR-Aware README says the SMID RAW data are transferred to RGB with the default ISP, while the Retinexformer README only says to process RAW to RGB and does not ship exact ISP parameters. The RGB conversion is therefore not independently reproducible from metadata alone.

## Remote low-input probe

The following paths were checked by existence/metadata only and were absent: `shared/t027a/data/SMID/SMID_LQ_np`, `shared/t027a/data/SMID/SMID_Long_np`, the placeholder SNR paths, and `/mnt/proj3/xgxu/EDVR/datasets/test_list.txt`. The only SMID artifact present was the 245-byte SNR `test_list.txt`; no low image payloads were enumerated. Reference paths are represented only as loader rules in the manifest.

## Accounting and reopening condition

`reference_reads=0`, `metrics=0`, `inference_runs=0`, `optimizer_runs=0`, and `low_payload_files_opened=0`. To reopen T072-G, provide an authorized processed RGB `SMID_LQ_np` payload and one shared test-list/conversion contract, then rerun the verifier. Do not claim a cross-domain benchmark or launch baseline inference from the current evidence.

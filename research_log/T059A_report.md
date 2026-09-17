# T059-A — exact detail Jacobian cache valid

**DONE: T059-A detail Jacobian cache valid.** All 7,346 canonical states pass the fixed chain-rule gate and zero-status checks. The separate-process finalizer passed before writing the authoritative completion marker. This is a reusable derivative-cache preflight only: no new energy was trained and no deployment/readiness claim changes the accepted T058-AF negative verdict.

## Fixed source and execution

Original authorization `3cb2854652a38245f567ae7da41f3e01c7ea206a`; execution review `af4d15acd84268a46f63b36ae97a6383a9e44add` accepted the same tested source `4ef5386b37cd79a1b0380632ac08b80fe398e47c`. All 207 Git-blob/source bindings verified before deployment and unchanged after the run. No implementation or scientific graph was modified following the execution review. PR #96; branch `codex/T059A-detail-jacobian`.

Sole physical GPU1/NVIDIA RTX A6000 run `20260917-165444-ttie-t059a-jacobian`, release `20260917-165416-ttie-t059a-jacobian`. Launcher start 2026-09-17T16:54:51+08:00, finish 17:18:57+08:00, exit0. Cache writer measured 1437.3296336699277 s; receipt completed 2026-09-17T09:18:52.030314+00:00. Torch 2.4.0+cu121, CUDA 12.1; exact environment and command in `T059A_run.sh`. Local6tests5.06s/server6tests1.44sPASS. No scientific rerun.

## Validation

| Check | Result |
|---|---:|
| Canonical rows, no filtering/gap/overlap | 7,346 |
| J shape/dtype | 28x64, float32 |
| Fixed allclose gate | atol=2e-6, rtol=2e-5; 7346/7346 PASS |
| Maximum elementwise absolute error | 1.7285346984863281e-06 |
| Maximum L2 error | 1.8820343504668232e-06 |
| Minimum nonzero cosine | 0.9999999999407484 |
| Exact-zero accepted/reconstructed rows | 98/98 |
| First16 T058-AC continuity | 16/16 PASS before index16 |
| Atomic deterministic chunks | 116 |
| Independent separate-process reopen | PASS |

Reconstructed gradients use the unchanged T01428-D grad-enabled feature forward and exact T054 Detail renderer. As in accepted T014 machinery, eight exact reverse-mode VJPs cover feature indices12-19. The other20 rows are exact constants in the new detail coordinate: active/signed/evidence and frozen legacy EV/gamma grid. They are not approximated. Independently evaluated frozen-head q at a detached copy of the same phi includes the existing head normalization/scaling; reconstruction uses CUDAfloat32 J^Tq. Each row matches immutable Stage-A canonical identity, raw/y0/active/detail/legacy tensor hashes before differentiation, and preserves them afterward. No direct learned-gradient recomputation substitutes for the accepted stored comparison.

All Stage-A shards/completion/receipt/manifest/chunks and every accepted tensor are verified before and after. All phi/J/q/reconstructed/accepted tensors are stored in chunks with per-row hashes; receipt records unchanged model/checkpoint/source-bank hashes. The separate finalizer reopens every tensor/hash and stored comparison, confirms exact canonical order, and writes complete.json only after success. An additional read-only CSV export replay reproduces all counts/maxima/minimumcosine and canonical identities; no numerical graph is rerun.

## Information boundary

`Image.open` denied throughout. Source-clean opens0; reference-gradient access0; optimizer updates0; persistent scientific-state changes0; FD/JVP0; target-domain/LOL-v2 access0; official-test access0. No AF reference gradients, PSNR/SSIM/oracle quantities, training/loss/weighting design, retraining, real-domain TTT, or renderer change. First16AC evidence is a learned-gradient verifier and contains no source-reference gradient.

## Evidence and recovery

Compact selection, manifest index, all7346-row CSV.gz, summary, continuity, receipt, writer reopen, independent reopen, completion, exact run/log, archive and export readback receipts accompany this report. The manifest index binds the original full manifest and all116 tensor chunks. Checkpoint manifest `complete:false` is a checkpoint field; the final authoritative `complete.json` binds its hash plus receipt and independent reopen.

Full raw tensors and before/after dictionaries: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t059a/T059A_evidence.tar.gz` and `/media/wenchang/F/wjq/TTIE/shared/t059a/T059A_evidence.tar.gz`; 11107825 bytes; SHA256 `9f17af620ccbba8afae699e1fe5d32a4d68a57003e77f17575a52c8556f61bde`; both copies verified identical. Raw manifest SHA256 `1dc8006aa6aef5fa4352cdce7a95124ca940633cc927ab9b18ede0fa2c85db19`. Complete marker SHA256 `3a607ccf88d7063a0b4fc4bb90ae5f150a39673315948618815b7078cc34ec10`. Large tensor storage on established server home/F is explicitly authorized.

Infrastructure issues: the first local export transfer encountered Ddisk-full and was stopped; its partial local copy is not authoritative. A local readback attempted before that transfer finished failed on a missing selection file, produced no verification receipt, and was replaced by read-only verification of the complete remote export. No scientific computation, source or tolerance changed. Compact evidence is published from remote project bytes through the local process/GitHub API without placing credentials on the server. Known NVML warning did not prevent CUDA execution. Initial Git fetch connection failure recovered on retry; PR history left unchanged.

Next: research-lead review of the validated cache. Stop this cycle; no T059-B, training or real-domain integration without a new task.

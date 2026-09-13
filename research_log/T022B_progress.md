# T022-B active

2026-09-13T19:17:54.989279+00:00

T022-A accepted/merged98054ad96d87f02ff2b6dea60a9199e0214b41f7. New task mainac930b4130ef88c2dee840ce76c4214197bc54ba. Branch codex/T022B-trajectory-headroom, sparse worktree .autodl/T022B-work.

Same100 validation split b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b and original run20260914-024025-ttie-t022a-core. Verify frozen input/trajectory/output hashes, then reconstruct all100 originally selected states BEFORE opening normal pixels, max error <=1e-6. GPU re-render of unchanged Region2 from saved raw state/gate only; no CLIP/head/Adam/trajectory calls. Stream4100 full-resolution PSNR/SSIM evaluations with original arithmetic. Report selected,41globalsteps,separatePSNR/SSIMoracles,headroom distributions/histograms,and EV/gamma saturation using frozen projection bounds1e-6. Independent aggregation. No official test/tuning/newTTT. Conclude material selector headroom exists or trajectory headroom is limited; structurally blocked if reconstruction fails.

Original remote root /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-024025-ttie-t022a-core/artifacts/audit. Low/normal inputs under shared/t022a/low and normal. Outputs contain11.8GB backing storage; preserve all originals and stream, no new image archive. Backup F/shared/t022a/T022A_execution.tar validated100outputhashes. Source is unchanged T014 code.

- 2026-09-13T19:22:21.653547+00:00: Four donor loader/metric tests pass11.16s; three reconstruction/oracle/saturation tests pass5.54s. Renderer/loader/metric Git blobs unchanged, static audit finds no TTT/Adam/CLIP/head execution calls. Ready to freeze and deploy streaming audit.

# T037-A: REFERENCE_DIAGNOSTIC_ONLY

Use accepted T036 cohort (279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b) and all frozen raw states. No optimization or fresh data. Reconstruct 8,200 states on A6000 GPU1 with exact accepted renderer; compare all selected outputs, identity states and physical fields before freezing image hashes. Only then score the same already-used normals with exact accepted PSNR/RGB-SSIM and independent numerical implementations. Original energy decisions remain unchanged. CPU parallel metric scoring preserves the accepted SciPy convention.

Classify strong late-selection headroom iff mean common reference-best minus selected PSNR >= 0.75 dB and at least 15 of the exact 29 prior PSNR-loss cases reach baseline selected PSNR at a strictly earlier common step. Otherwise limited/mixed late-selection headroom. Report SSIM in parallel, all step curves, loss cases and worst low00559 trajectory. No stopping rule, new cohort, official test or deployable changes. Stop after report/PR/mailbox.

Local focused diagnostic tests: 2 passed (1.95 seconds). Frozen inputs are accepted T036 artifacts; normals remain outside reconstruction CLI. All retained per-step images stay on the server with F backup; only scalar evidence is fetched locally.

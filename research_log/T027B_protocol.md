# T027-B fixed native-pad16 exporter smoke

Task inbox920a472d, main9f2d1f51. One integration/provenance task; stop after
eight-image parity, no quality metric, validation/test inference or Ours change.

Canonical official source: JIA-Lab-research/SNR-Aware-Low-Light-Enhance at
1113144c82adc8bcc4a9ec27749ed75f196a4e4d. Verified53 Python/YAML/README blobs
against the pinned Git tree. No project-level license grant: third-party
source/archive/checkpoint bytes stay outside Git. Only TTIE-authored code,
hashes, locators, commands and experiment receipts are submitted.

Official Drive1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb is a1017280510-byte checkpoint
ZIP. Downloaded only its designated `pretrain_model/LOLv2_real.pth` member using
official HTTP206 byte ranges. Decompressed size156523164bytes, CRC32 1209377b,
SHA256432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781.
Exact ranges, compressed hash, ZIP tail hash and locator are preserved. Server
Google Drive network is unreachable; local official-member download followed by
workflow SCP preserves identity without mirrors. All checkpoint keys strictly
match the official network. No weight changes are permitted.

## Reuse and fixed preprocessing

Independent path imports official `networks.define_G`, official
`Video_base_model4_m.VideoBaseModel.test`, and `data.util.read_img_seq`.
It extracts the official dataset's four low-only blur assignments at runtime
without persisting their source. No original dataset constructor, target reads,
visualizer, metrics or `test4` is called. The wrapper supplies only netG, var_L
and nf. Official missing config center maps to None, preserving factory default.

The TTIE exporter independently imports the same unmodified architecture,
loads the same strict state, decodes native RGB float32/255, constructs the
native5x5 OpenCV blurred feature and low-derived SNR, calls netG and saves HWC
float32 before quantization. It does not call the independent adapter or official
test method. Shared task audit only covers image access, not numerical work.

Fixed order: native RGB -> official5x5blur at native resolution (OpenCV default
REFLECT_101) -> right/bottom reflection-pad both low and blur to multiple16 ->
SNR from padded low/blur -> official network -> unpad -> clamp[0,1]. For400x600,
padLRBT=[0,8,0,0], network receives400x608, then output returns400x600 exactly.
No input resize/interpolation. Internal official network math is unchanged.
SNR uses RGB weights .299/.587/.114, absolute low-minus-blur luminance noise,
signal/(noise+.0001), per-image max normalization with+.0001, clamp[0,1].

This is the predeclared `ttie_native_pad16` adaptation, **not** exact reproduction
of official resize-based `test4` inference or published numbers.

## Cohort, tests and execution

Subtract the frozen100 validation filenames from689traininglow names; select
eight of589 by ascending SHA256(`TTIE-T027B-smoke|<relative-low-path>`).
Read ZIP directory names only and extract exactly selected low members. Each
real low once per independently wired path; no paired normals, validation or
test image decoding/scoring. Both processes enforce exact8image read allowlists.

Baseline release20260914-155401-ttie-t027b-baseline,
run20260914-155420-ttie-t027b-baseline: PASS8native finite clamped outputs,
unchanged parameter hashes, exit0. Then exporter focused tests: local3PASS16.99s
(native8column reflection/unpad, known SNR including dark-zero case, CLI surface).
Next: same tests remotely, one8image exporter run, compare each saved float
array to baseline with max absolute error<=1e-6, no tolerance changes or tuning.

Separate generated32x40 synthetic tensor, four absent/mutated/withheld target
canary conditions, both actual pretrained paths. Deny attempted canary reads;
hashes must stay invariant. No real smoke low repeated for the counterfactual.
Check network state hashes before/after both real paths and synthetic check.

Report per-image errors/hashes, decoded paths, source/weight/environment
bindings, native pad/config, mean/median/p95 runtime and peak memory. Timings
start from decoded CPU tensor, include native blur, CUDA transfer/pad/SNR/model
and float CPU output; include first-call initialization, exclude decode/load.
No baseline ranking or cross-method speed/quality claim.

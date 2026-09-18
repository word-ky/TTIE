# T059-W active-path online replay

Classification: `active-path online target-free Jacobian bridge is reproducible across the fixed outer source images`. All 16 selected anchors have positive cached and fresh online gradients and active regions; all 16 pass unchanged V bounds. This is source numerical reproducibility, not safety or enhancement quality.

Authorization 0098138c8d074f0d39d6196599e74333936cc1ce; source 5e25e9d76589f4f733b0d11bb0badd0c7259c0e6; inherited V b4bd9ee9e2df829b915a0494c50ed1642bcfa102; U evidence 88de78b1a91adfc6dec71c378cdf8a4ec044cb51.

Selection uses only U pre-reference actions.json/action_freeze.json metadata: fixed16 source images, maximum predicted norm per image, smallest bank tie. Norm is not a confidence gate. Selection fsynced/hash before online run; SHA256 061734caba7d3dc95f79434cb18629c0f925ac5c08de502801901188e3195f27. No image substitution.

Exact unchanged V numerical path: image-only frozen CLIP/prototypes/gate, original28-D features and fresh autograd J, fixed E head/train normalization, J-transpose-q, literal T0548x8 shared renderer and one fresh Adam lr.05. Selected state0 y0/masks match U exactly. Source222 bindings and all inputs immutable. All model computation on physical A6000GPU1; q retains CPU E convention.

Head SHA e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0; source/checkpoint/feature/renderer exact hashes in selection/source/onlinefreeze. No cached field or action tensors read by online process; cached comparison only after all16 bundles persisted/fsynced/hashed.

Online freeze 2026-09-18T19:11:22.184896+00:00; first cached tensor read 2026-09-18T19:11:24.720790+00:00. All forbidden counters zero: clean/reference images, reference gradients, target-domain, LOL-v2, official test, inference reference leakage; no reference MSE/PSNR/SSIM computed.

Max/min errors: {"x_max": 1.2874603271484375e-05, "J_relative_max": 1.3895649269392029e-06, "g_cosine_min": 0.9999999999832085, "g_relative_max": 6.882436190954023e-06, "y1_max": 5.960464477539063e-08, "v1_max": 5.587935447692871e-08, "q_max": 8.940696716308594e-07, "active_anchors": 16}.

|Image|Bank|Selected norm|Active cached/online|x maxabs|J relF|g cosine|g relL2|y1 maxabs|Pass|
|---|---|---|---|---|---|---|---|---|---|
|36660|1|0.0840701289068|4/4|3.57627868652e-06|1.1241434637e-06|0.999999999998743|1.79313728312e-06|2.98023223877e-08|True|
|37670|26|0.0752938827161|4/4|2.50339508057e-06|1.08149384519e-06|0.999999999998893|1.51528354639e-06|2.98023223877e-08|True|
|38070|54|0.0962142638051|2/2|2.86102294922e-06|6.58736548429e-07|0.999999999999568|9.299562282e-07|2.98023223877e-08|True|
|38825|77|0.480368654774|4/4|2.86102294922e-06|1.28983056358e-06|0.999999999998985|1.58450398893e-06|5.96046447754e-08|True|
|39551|102|0.298214365016|4/4|6.19888305664e-06|1.38956492694e-06|0.99999999999821|1.9604692538e-06|5.96046447754e-08|True|
|39951|127|0.260519339496|3/3|2.62260437012e-06|9.81099013272e-07|0.999999999999134|1.4918170926e-06|5.96046447754e-08|True|
|41488|152|0.153219259494|4/4|5.24520874023e-06|6.74746883258e-07|0.999999999998063|1.97318332426e-06|5.96046447754e-08|True|
|41990|177|0.34771611182|4/4|1.1682510376e-05|7.5895091814e-07|0.999999999999048|1.38738509151e-06|5.96046447754e-08|True|
|42563|204|0.295062177549|3/3|1.28746032715e-05|6.34210053767e-07|0.999999999983209|6.88243619095e-06|5.96046447754e-08|True|
|43435|227|0.175085431169|4/4|2.38418579102e-06|1.06657778028e-06|0.999999999999421|1.18369955411e-06|5.96046447754e-08|True|
|44195|252|0.414105304057|4/4|8.70227813721e-06|9.37347759542e-07|0.999999999999476|1.04220828425e-06|5.96046447754e-08|True|
|45070|277|0.162422007061|4/4|3.81469726562e-06|9.47328483424e-07|0.999999999998344|1.85549274589e-06|5.96046447754e-08|True|
|45728|302|0.148693493188|4/4|2.62260437012e-06|1.19895678102e-06|0.999999999999308|1.18999124865e-06|5.96046447754e-08|True|
|46463|327|0.118688117104|4/4|2.26497650146e-06|1.11899892361e-06|0.999999999999617|8.84634384109e-07|5.96046447754e-08|True|
|47121|352|0.201015304548|4/4|1.14440917969e-05|8.88032637644e-07|0.999999999999555|1.2830064977e-06|5.96046447754e-08|True|
|47801|377|0.357083979776|4/4|3.93390655518e-06|8.53441075403e-07|0.9999999999995|1.01180930119e-06|5.96046447754e-08|True|

Validation:9 focused tests pass2.70s. Independent verifier separately reconstructs maximum/tie selection, all numerical errors, J-transpose-q, firstAdam, SciPy renderer, fixed bounds and hashes;16PASS.

Sole online/comparison run 20260919-031106-ttie-t059w-active,19:11:10-19:11:27UTC: online and comparison completed, process exit1 at independent verifier import. Local select.py shadowed standard-library select during SciPy import. Recovery changes only launch to `python -m research_log.T059W.verify`, preserving source and all frozen tensors. Verification-only run20260919-031158-ttie-t059w-verify-only,19:12:03-19:12:09UTC,exit0. No online/action/comparison rerun, tolerance change, differentiation change or scientific failure. Use module invocation for replay.

Other pre-run infrastructure issues: authorization-transfer shell quoting corrected with heredoc; source-manifest regular-file enumeration corrected after __pycache__ directory error; one SSH timeout before source upload recovered. D full and NVML warning persist, actual CUDA execution succeeds. No dataset/checkpoint relocation or driver modification.

Raw evidence archive {"home": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t059w/T059W_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t059w/T059W_evidence.tar.gz", "sha256": "0b75df4a603ea369999dd692d395c5a298f78992df468175a7c92e53a9ba1be0", "bytes": 38913170, "home_verified": true, "F_backup_verified": true}. Compact tensor fields and full16 errors are Git-delivered; full images in verified home/F archive. Next: research-lead review; no target-domain rollout, second step, retraining or self-merge.

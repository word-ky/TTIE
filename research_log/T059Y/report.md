# T059-Y fixed T026 integration

Classification: `one-step matched-detail integration is practically immaterial on the opened development cohort`. Mean PSNR gain +0.0024241736224814583dB fails the fixed +0.05dB materiality gate. Stop the integration line without tuning. Validation-only, no deployment/SOTA/safety claim.

Authorization ed725b006a674521856fd85f982523fd0744df91; source e243c5dc0914a7745172fc3cf1e079bbbfb8b26b. Exact100 Xdevelopment lows/order; accepted T026run20260914-113853-ttie-t026a-gamma05 and evidence d577fc24a54cdb0e3de22bc3d07dd14168c10e70. Accepted pre-reference freeze Git blob62904f6907e80cab5cc851a3355afc9bca99a90c verified. All300 output/decision/trajectory hashes verified before execution. No baseline rerun or reselection. Final cohort SHA b36a8ce4105d83a33b6d752c5e52f6c1b3223e4e0d79750419f9787af2be5e1e.

Each exact selected T026image is detailbase; its selected raw/grid match original trajectory at persisted selected_step. Gate/active/winner/evidence/calibration copied from accepted target-free inference records; no gate recomputation on enhanced images. Fixed grid enters unchanged28-D features. Independent verifier checks feature first12 against original gates and last8 against selectedgrid. Only v startszero and receives exactlyone Adam lr.05. T026raw/grid remain immutable and match finalsaved global state. No cachedmatched-detailx/J/q/g/action reused. Fresh CLIP current-image features/J, frozen Ehead e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0/trainnormalization, exactT0548x8sharedrenderer. A6000GPU1,TF32off,seed7.

## Frozen gates

|Quantity|Observed|Requirement|Pass|
|---|---:|---:|---|
|Active detail actions|100/100|>=50/100|yes|
|Mean PSNR gain|0.0024241736224814583dB|>=0.05dB|no|
|Median PSNR gain|0.0008808606100778604dB|>0|yes|
|MSE improvements|91/100|>=60/100|yes|
|Mean SSIM gain|0.0014076023766469731|>=0.001|yes|

Improve/harm/tie91/9/0 at fixed signed1e-12 rule. Maximum MSEharm3.482472213692765e-5. No exclusions.

|Metric|T026 mean|Integrated mean|Mean delta|Median delta|p10 delta|p90 delta|
|---|---:|---:|---:|---:|---:|---:|
|mse|0.0982122937939268|0.0981687609538827|-4.35328400441478e-05|-1.91627885299736e-05|-5.7664668902975e-05|-1.89141407702391e-07|
|psnr|11.1208764173496|11.123300590972|0.00242417362248146|0.00088086061007786|3.55696156578489e-05|0.00405755844283497|
|ssim|0.373791825151708|0.375199427528355|0.00140760237664697|0.001212865242793|0.000199033128167521|0.00266646393932949|

All100 per-image values in evaluation_table.json; norms/active counts/gates/state and output hashes in online_records.json. Harm rows: [{"index": 5, "low": "Train/Low/low00509.png", "norm": 0.08530706467168252, "mse_change": 6.6247418287618265e-06, "psnr_change": -0.0007681882269796603, "ssim_change": 0.0013019317761905902}, {"index": 8, "low": "Train/Low/low00351.png", "norm": 0.07078467638167664, "mse_change": 3.42910856266157e-05, "psnr_change": -0.0025559653291278295, "ssim_change": -0.00021648507518556848}, {"index": 14, "low": "Train/Low/low00080.png", "norm": 0.09831430159745086, "mse_change": 2.42904064733096e-06, "psnr_change": -0.00012708775641279146, "ssim_change": 0.00028014950096411084}, {"index": 15, "low": "Train/Low/low00501.png", "norm": 0.09372791753336993, "mse_change": 2.4303660668203997e-06, "psnr_change": -0.00023034277968392303, "ssim_change": 0.0018190523989574103}, {"index": 18, "low": "Train/Low/low00206.png", "norm": 0.05135500643795559, "mse_change": 6.209177368064756e-07, "psnr_change": -0.00020007614457639988, "ssim_change": -0.0005790899331747923}, {"index": 37, "low": "Train/Low/low00563.png", "norm": 0.09571320760491851, "mse_change": 3.482472213692765e-05, "psnr_change": -0.011313964557590594, "ssim_change": -0.0019426442442974645}, {"index": 44, "low": "Train/Low/low00510.png", "norm": 0.12334188426238148, "mse_change": 1.25251599498305e-05, "psnr_change": -0.0004980508965530817, "ssim_change": 0.0005772589433399311}, {"index": 45, "low": "Train/Low/low00562.png", "norm": 0.07037381694191297, "mse_change": 2.8466459423337476e-05, "psnr_change": -0.0013276914721540578, "ssim_change": -0.0003843992711939359}, {"index": 60, "low": "Train/Low/low00505.png", "norm": 0.11668336986621941, "mse_change": 1.7613443117667216e-05, "psnr_change": -0.001089542905379659, "ssim_change": 0.001758186700005826}].

## Boundary and validation

Inference start 2026-09-18T21:35:12.138918+00:00;global100freeze 2026-09-18T21:35:37.195080+00:00;first developmentnormalread 2026-09-18T21:35:42.094726+00:00. Alloutputs/decisions fsynced/hashed before evaluator starts. Inference has no normalroot argument; filesystem audit allows only pinnedcode/model/low/T026inference artifacts/outputdir. No Xper-image outcomes or evaluation files. Original247 sourcebindings,allinputs/outputfiles/head unchanged. Inference clean/normal/referencegradients/matched-detailcaches/metricfiles/officialtest/leakage0. Developmentraw100 bound and hashed,0decoded;T026base100 decoded;T026globalupdates0;detailsteps100;headtraining0. Normal evaluation100 plus independentverification100 only afterfreeze.

7focused tests pass4.18s (firewall,input mutation,state mutation,gates,original T054 math). Independent100PASS checks exactbase/globalstate/selectedtrajectory/gatefeaturecoordinates,NumPyJ-transpose-q/Adam,SciPyrenderer,independentSSIM/TorchMSE,quantiles/counts/classification and immutablehashes. Maxmetricerror 7.105427357601002e-15. Metrics exactly inheritedRGBGaussian11sigma1.5reflectpopulation,no crop/resize,float64 arithmetic over nativefloat32 pixels;PSNRfloor1e-12.

Sole run20260919-053506-ttie-t059y-integrate exits0 at21:36:18UTC,2026-09-18. No execution failure,repair,rerun or scientific deviation. One exploratory missing-filename lookup was resolved before implementation;no experiment effect. Dfull/NVMLwarning remain nonblocking. Results match accepted T026mean11.120876417349557dB/0.3737918251517076SSIM.

Result SHA a0f05f20b53d20f7a4f3a857d151a1e384ef50747e02e66c3cfae82c5fc14039. Full rawarchive {"home": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t059y/T059Y_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t059y/T059Y_evidence.tar.gz", "sha256": "da7ce1c99e1a53a18734cee08189912287bc88abe0c97a5d8100f93f4a0a9aa1", "bytes": 356895724, "home_verified": true, "F_backup_verified": true}. Fullimages remotehome/F;compactfields and tables Git. Next: close this fixed one-step integration line for lead review;no tuning,secondstep,T036integration,targetfitting,officialtest or selfmerge.

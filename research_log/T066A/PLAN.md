# T066-A fixed19D dynamics safety audit

Authorization0eb6086524fc18cb9ffee0b9606fcfc1e1bb5c27; exact OPEN task in authorization.md. No historical merges.

Reuse exact T065-A11snapshot features and original mean/scale. Append only the8specified temporal quantities:3stepobjective drop, objective curvature, RGB step/acceleration RMS, raw12state step/acceleration L2, mean-luminance step absolute change, gradient-ratio step absolute change. Full0..27prefix; missing history0, obj_drop_3 explicitly usesmax(0,k-3). Acceptedfloat32GPU renders/objective, float64GPU temporal differences/reductions and float64CPU progress.

LOIO interpretation fixed before results: old11normalization remains original as expressly required; each fold's new8mean/population-std is calculated only on its99training images, with scale floor1e-8. This prevents held-out-image statistics entering the new fitted normalization. For final fit (only if permitted), new8normalization usesall100development images. Concatenate unchangedold11andnew8normalizations into19D.

Reuse T065-B class-balanced Newton solver with only dimension enlarged19coefficients+intercept: weightsN/(2N_class), summed weighted logistic NLL+.001||w||^2/2, intercept unregularized, zero start, logitsclipped[-30,30], max50updates, early stop onlystep_inf<1e-12. No sweep, alternate solver or post-outcome retries. Save each of100fold models/normalizations, training-image membership and complete Newton traces. Independent verifier recomputes each fold, features and labels, checks Newton equations and all probabilities/choices.

Aggregate held-out state confusion and unsafe recall; evaluate per-image rollback guard at frozenrho checkpoint. If recall<.50 or anyfivegatefails, report DEVELOPMENT_DYNAMICS_NEGATIVE and stop without full-data model or transfer quality reads. Otherwise fit identical rule once onall100development images, freeze model/normalization/source/state/rule, freeze100transfer choices/outputs, then read reference quality. No optimizer reruns, newcohort, officialtest or crossdatasetaccess.

Reuse renderer/data scope/source binding/metrics from accepted T065-B. Development T026/T036 anchors are accepted Git-bound evidence; independently recompute state PSNR/SSIM. Independentfeatures NumPy, independentNewton SciPy/einsum. Primary render/featuresGPU1; smallfloat64LOIOsolvesCPU as authorized.

Tests:16baseline passed5.81s;3dynamic/fold/19Dtests passed4.41s;4including independent temporal/fit comparison passed8.33s. Full affected tests and one committed GPU audit follow.

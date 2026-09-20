# T066-B predeclared diagnostic

Reuse the exact T066-A artifact feature tables, final model and all-development 19-D normalization. Development nearest-support LOIO excludes all 28 query-image states but keeps this same frozen all-development normalization, as the diagnostic explicitly fixes the feature space (no fold refitting or renormalization). All-development model probabilities and the accepted LOIO classifier summary are distinguished.

Freeze target-free development/transfer tables and recomputed frozen-model probabilities before any transfer reference-quality read. Bind prior features/model/normalization/freeze and source by SHA-256. Then rerender frozen transfer states on GPU and compute offline PSNR safety labels; references never affect representation or probabilities. No optimization/model fits/selection changes.

GPU float64 ordinary Euclidean distance, batches of 28 queries, nearest single safe and unsafe development states. Deterministic within-class tie break is original flattened image-index/step order. Strict margin>0 defines safe-like support; report exact ties separately. Development geometry excludes entire query image. Distribution uses min, linear-interpolated quartiles, max and mean. Classify using exactly the four authorized criteria.

Independent verification reuses the accepted renderer and independent T066-A NumPy feature reconstruction, independently computes frozen probabilities, CPU reference PSNR, SciPy Euclidean distances, masks/confusion/distributions/category and hashes/read ordering. No new classifier/guard or qualification claim. No transfer/new-cohort/held-out tuning.

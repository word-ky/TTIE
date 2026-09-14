# T033-A fixed benchmark execution

Task from main3ca68892; T032 accepted mergefbff9c04. One100-image original validation pass with unchanged T027-A exporter and pinned official Retinexformer checkpoint; no Ours rerun/tuning, SNR run, new split, perceptual metrics or official test. GPU1 preferred.

Reuse `ttie.retinex_exporter.main` verbatim under a low-image allowlist audit wrapper; preserve original config/network, strict loading, OpenCV float32 RGB/255, factor4 native padding, clamp, inference mode and timings. Before inference bind original split SHA b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b, accepted six official source files/archive, official checkpoint6478393bytes SHA539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b and exact exporter/config bytes. All100 outputs and decode receipt freeze before separately bound evaluation authorization. Existing normals elsewhere on server are outside inference allowlist, not claimed absent.

Reuse accepted T026 evaluator's pixels and independent SSIM plus unchanged rgb_ssim. Crucially normal pixels round tofloat32 beforefloat64 arithmetic, unlike the later T030 fresh evaluator. Accepted Ours100 metrics read-only CSV SHAad704fca9dbc393a0e30737eae60d212d41bb797e630e6599d39059380e5a397. Pair by exact low/normal/order, no Ours output/model rerun.

Baseline tests3passed14.67s before wrapper changes. Focused unchanged exporter geometry/API tests plus external freeze replacement rejection then actual GPU end-to-end100. No quality/promotion threshold; report benchmark complete on mechanical acceptance. Full image outputs remain server/F backup; compact receipts/metrics local due D disk<80MB free. Maintain original official-source blobs and existing license; no upstream modification.

Interpretation: this is the requested development-split anchor. The split was carved from official LOL-v2 training pairs, while the official external checkpoint is the LOL-v2 Real supervised checkpoint; do not represent the comparison as an untouched held-out evaluation of the external model or a SOTA result.

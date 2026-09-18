# T059-S fixed implementation plan
Authorization: 75af50ce71e78ddb8b8a3641700fbe14afb88e69. Accepted E source 830e80ba0e1a4d09bce9c9ffd57be162a781d013; T054 source f4baa579e4441a6edf5ec818ddca87f28f1b7e5d.

Use literal accepted T054 coefficient/forward and optimizer assignment extracted from its hash-pinned source. T058 constructor supplies the canonical source y0/detail/mask; all80 full constructor states already match accepted T059A. No target-domain oracle is imported or executed. Synthetic unit test compares injected-gradient first Adam step to original optimize(...,steps=1) trajectory before its best-step restoration.

The accepted E detail statistic computes CPU derivatives on its full1529-row batch. Preserve that batch exactly, select its80 anchors and execute each fresh Adam step/render on physical GPU1. No new CLIP feature forwards. Feature-vs-recomputed-phi differences exist only in17 inactive banks; accepted E cache convention is unchanged.

Eligibility is inherited E reference-gradient norm>1e-12 and is attached after all80 action freezes. All80 anchors receive one optimizer step. Gates use eligible anchors; all-bank and per-image statistics also reported. Relative MSE=(mse1-mse0)/mse0. For exactzero anchor MSE require output MSE alsozero and reference-ineligible, then report relative0. Descriptive PSNR uses a1e-12 floor. Harm=max(relative change,0), p90 uses linear quantile. No tuning.

Full1529 predicted-gradient hash must match accepted E detail statistics. Anchor-only reference byte ranges must match accepted E selected storage receipts. Then original E detail_statistics on the same batch with anchor-only mask must exactly reproduce direct anchor fraction/median. No outer reference storage decoded. Source JPGs are limited to16 pinned inner-held images, only after freeze.

Separate processes: act -> evaluate -> independent verify. Source hashes frozen before execution; all80 output tensors/fsync/hash before references. Large image tensors remain in dual-disk project archives with full manifests; compact source/receipts/results go to GitHub. No second action run or step permitted.

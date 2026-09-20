# T065-B fixed class-balanced linear safety rollback

Authorization d442a47e7a4f97d56f20507581f7b69573e36583, exact OPEN task in authorization.md. No historical artifact merges; engineering branch evidence and mailbox append only.

Reuse: import exact T065-A feature function/feature order and accepted normalized-progress selector. Reuse T065-A model.json mean/scale verbatim (hash bound), not refit normalization. Frozen T062-A development and T063-D exposed transfer states and low PNGs, no Adam rerun. Existing GPU render and post-freeze evaluator/read-scope machinery retained.

New bounded logic: safe=1 iff development PSNR(state)-PSNR(T026)>=-5.614. Report counts before fitting; stop INSUFFICIENT_CLASS_SUPPORT if either absent. Weights N/(2N_class). Zero initialize12coefficients including intercept. Float64 CPU Newton/IRLS, exactly sum weighted logistic NLL +.001*||w||^2/2; intercept unregularized. At each iteration clip A beta to[-30,30], compute sigmoid, gradient and IRLS Hessian, solve H step=g, update beta-=step. At most50updates; early stop only maxabs(step)<1e-12. No line search, alternate convergence rule, sweep or retry. Preserve per-step loss, before/after coefficients, step and infinity norm.

Inference: fixed p_safe>=.5; keep base if safe, otherwise latest earlier predicted-safe state, else0. Development confusion matrix uses safe/unsafe rows and predicted-safe/unsafe columns; unsafe_pred_safe is the unsafe->safe miss count. Require allfive development gates or stop DEVELOPMENT_NEGATIVE before transfer. If passed, freeze model/source/normalization/training table before target selection, then freeze100choices/outputs before reference-quality reads. Report five transfer gates, rollback count/histogram and post-freeze tail16/86.

Independent verifier: recompute GPU renders and all11features with accepted independent NumPy implementation; recompute development labels and class counts; independently solve logistic Newton with SciPy/einsum and check each saved Newton equation/update/stopping condition. Verify normalization/hash, probabilities/choices/read ordering/metrics/classification. Reuse accepted Git-bound T026/T036 development metric anchors, recompute all state PSNR/SSIM; independently score transfer controls.

Tests:11baseline passed16.01s;4classifier tests passed8.32s;5including independent fit/trace equations passed8.17s. Single-thread sequential MKL used on Windows as established. Full affected tests and one source-bound GPU run follow. No method changes based on outcomes.

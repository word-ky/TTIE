# T071-B matched official baseline evaluation

Authorization: main ab9e7df5c268f965bca2949b0f24ff51d321d8c0 (T071-B inbox).
Scope: exact frozen T033-A Retinexformer and T045-A SNR-Aware, all 100 T071-A official pairs. No Ours inference, retraining, tuning or cross-dataset access.

Reuse map: unchanged ttie/retinex_exporter.py and snr_exporter.py main functions own all model loading, preprocessing, inference and postprocessing. T071A/core.metrics owns metric math. scripts/evaluate_t026a.independent_ssim owns independent SSIM. Existing baseline tests cover padding, clamp, SNR and input interface. New wrapper only substitutes exact official low list and records freezes; postfreeze evaluator adds paired differences against saved Ours.

Increments: baseline tests -> wrapper/paired-summary tests -> immutable source deployment -> two sequential GPU exporter processes -> postfreeze metrics -> independent verification -> archive/report.

Training: both official LOL-v2 Real supervised checkpoints are bound to published paired Train/Low and Train/Normal recipes. Retinexformer recipe uses Test/Low and Test/Normal for validation; report that explicit upstream exposure, without claiming independent unseen test status for that checkpoint or knowledge of actual checkpoint selection. SNR training recipe uses Train for validation. No training in this task. SNR uses previously accepted native_pad16 protocol adapter, not official resize-test4 paper numbers.

Output stage: only staged T071-A low files may be decoded; no archive or quality table is consumed by exporters. Each complete 100-output table is hashed before post-hoc reference metrics. Model/global source hashes checked before and after. Native float outputs retained, no quantization. Same T071-A RGB float64 PSNR and 11x11 Gaussian SSIM.

Timing: exporter-synchronized inference runtime; excludes disk decode/write/model load. Retinex timing starts after low H2D; SNR includes low-derived blur/SNR/H2D. Ours timing is the accepted API measurement. These are documented method-level timings, not identical service-latency benchmarks.

Stop after matched table or a task-specified blocker. Frozen weak numbers are reported without tuning.

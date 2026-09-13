# T021-A preparation and clarification request

Status: PARTIAL — exact comparison baseline and acceptance pool need research-lead binding before real SSIM evaluation.

Accepted T014 PR14 merge cebecffbd1335fade336df17d653eb4e5fb65ba3 records several baseline methods, not one H0: identity; region2_direct; region2_discrete_projected; fixed_step_source (16); global/bilinear Sobolev controls; and matched region2_ttt_energy_value_only. Its clean/homogeneous/heterogeneous gates use different comparisons. T021-A asks to reuse the exact accepted H0 but does not name which method to compare. Choosing one changes the scientific question and bootstrap result.

The accepted fresh run has40 source images and240 rows:200 across five primary conditions and40 offset_left_right_40 rows explicitly marked report-only by the original qualification. Heterogeneous primary improvement pools80 left_right/quadrants rows. Please specify the primary bootstrap pool (200 primary rows,80 heterogeneous rows, or all240 including offset) as well as the exact H0 method. No real SSIM score/CI has been computed or inspected, and no method/component has been rerun.

Completed preparation:

- Recovered immutable accepted manifest/config/output-verification/final-check origins and original source definitions.
- Hash-verified all480 episode output/decision files (240 rows,40 source images,3924677163 bytes) at the original T014 Stage-B F storage root. No tensor deserialization or clean-reference pixel access. Every evaluated candidate control and T014 output is recoverable from these previously frozen outputs.
- Clean reference can be recovered from the same source image's frozen clean-condition identity output: the accepted source applies unit gain and clamp to an already clamped clean tensor; this is not a new degradation/render/resize. This linkage will be finalized for the selected triples only after comparison scope is specified.
- Implemented fixed float64 RGB Gaussian11×11/sigma1.5/population SSIM over the full unresized image, with symmetric reflection borders and no crop. Cross-check uses skimage0.24 full SSIM map mean; its default scalar drops border pixels, so it is not used for this no-crop task.
- Implemented seed7 NumPy PCG64 10000 image-cluster bootstrap, preserving all rows and repeated sampled clusters, two-sided percentile95% with linear quantiles. No fresh data were used in tests.
- Two tests passed1.01s: identity/controlled perturbation and edge-sensitive full-map cross-check; clustered sampling with unequal row counts and multiplicity.

No scientific verdict yet. This is not structurally unsupported: original outputs exist and match hashes. Await the research lead's explicit H0 and primary pool. No need to repeat recovery or tests when only these bindings are supplied.

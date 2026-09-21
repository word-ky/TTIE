# T069-A exact final-transition projection pressure

Reuse T068D source2c33a93377d69f83727e847faed20add5ff38e24 endpoint/input/freeze/label/verifier flow; last completed9tests local16.17s/remote1.86s and4015GPU states PASS. Internal same-repo code, no new dependencies. No optimization or fit.

Raw states have shape[steps,1,3,2,2]; select state[k][:,:2] and pre_box[k-1][:,:2], retaining all nodes and excluding common gain. Compute float64 norms, exact1e-12 floor and zero cases; no extra clipping/weighting. Hash full previous/proposal/end tensors plus recorded bounds and input trace. For k0 no proposal exists; previous/proposal arrays/hashes null, endpoint bound, norms/score0 with no_transition reason.

Increment1 index/slicing/zero/ratio/nearest-rank tests. Increment2 minimal adaptation of diagnosis flow; affected tests; source commit/push; real server run and independent verifier. Primary CPU for eight-coordinate norm; GPU verifier replays the actual frozen CommonBox on proposal copies from both cohorts, reconstructs box from recorded active/winner and matches recorded lower/upper exactly, compares resulting EV/gamma to recorded state[k] exactly. Rerender each fixed endpoint on GPU and verify output hash. Independent norm via math.fsum over coordinates, strict flags and T99; score tolerance1e-12, projection exact. No changes to original trajectory.

Development100score/T99 nearest rank sorted[98] then transfer100score freeze before label hashing/joins. No development reference quality. Post-freeze T067C endpoint safety labels crosschecked T066B prior labels; no new reference metrics. Descending ranks1+strictly greater; percentile fraction<=score. Stop after diagnosis, no guard/rollback/window/sweep. All fresh/final sets sealed.

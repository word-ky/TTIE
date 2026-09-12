# T008 fixed restoration pilot

Qualified: False; failed: ['clean_p95', 'spatial_vs_global', 'beyond_direct', 'bright_region_safety']

| Condition | Method | Mean MSE | Median MSE | P95 MSE | Mean finite PSNR | Mean recovery |
|---|---|---:|---:|---:|---:|---:|
| clean | identity | 0.00000000 | 0.00000000 | 0.00000000 | null | null |
| clean | global_direct | 0.00064564 | 0.00000000 | 0.00168891 | 22.46769489 | null |
| clean | spatial2_direct | 0.00075089 | 0.00000000 | 0.00329086 | 20.98551172 | null |
| clean | global_discrete | 0.00131971 | 0.00000000 | 0.00774395 | 18.23057573 | null |
| clean | spatial2_discrete | 0.00167433 | 0.00000000 | 0.01432508 | 16.76274372 | null |
| clean | global_ttt | 0.00055706 | 0.00000000 | 0.00282581 | 22.05000560 | null |
| clean | spatial2_ttt | 0.00081690 | 0.00000000 | 0.00928912 | 19.65833309 | null |
| homogeneous_dark | identity | 0.08077710 | 0.07898793 | 0.13144569 | 11.35691821 | 0.00000000 |
| homogeneous_dark | global_direct | 0.05438458 | 0.04761959 | 0.10468828 | 13.24795283 | 0.32976996 |
| homogeneous_dark | spatial2_direct | 0.05422212 | 0.04864798 | 0.10446577 | 13.22084520 | 0.32578749 |
| homogeneous_dark | global_discrete | 0.01728456 | 0.00278198 | 0.08773045 | 22.78465172 | 0.80332721 |
| homogeneous_dark | spatial2_discrete | 0.02672991 | 0.01010310 | 0.10184794 | 19.39058718 | 0.70379826 |
| homogeneous_dark | global_ttt | 0.02137304 | 0.01127846 | 0.08773045 | 19.47061226 | 0.72555924 |
| homogeneous_dark | spatial2_ttt | 0.03129388 | 0.01653374 | 0.10345463 | 17.81463880 | 0.64067215 |
| homogeneous_bright | identity | 0.03899845 | 0.03869122 | 0.05983265 | 14.31731069 | 0.00000000 |
| homogeneous_bright | global_direct | 0.01292278 | 0.01043948 | 0.03256719 | 19.85865494 | 0.62008271 |
| homogeneous_bright | spatial2_direct | 0.01701866 | 0.01499925 | 0.03640148 | 18.42616543 | 0.50572875 |
| homogeneous_bright | global_discrete | 0.01339382 | 0.00936901 | 0.03828698 | 21.19280103 | 0.53663428 |
| homogeneous_bright | spatial2_discrete | 0.02405704 | 0.02294455 | 0.04515971 | 16.83667488 | 0.31115479 |
| homogeneous_bright | global_ttt | 0.01472249 | 0.01397495 | 0.03146702 | 19.09565914 | 0.57671705 |
| homogeneous_bright | spatial2_ttt | 0.01610558 | 0.01414750 | 0.03599500 | 18.63804764 | 0.55035676 |
| left_right | identity | 0.05999701 | 0.05663328 | 0.09583770 | 12.51815365 | 0.00000000 |
| left_right | global_direct | 0.06045163 | 0.05837812 | 0.09621639 | 12.47652924 | -0.01252935 |
| left_right | spatial2_direct | 0.04074756 | 0.03782693 | 0.06777459 | 14.22907949 | 0.30615093 |
| left_right | global_discrete | 0.06523331 | 0.06291009 | 0.11826038 | 12.22876594 | -0.14145578 |
| left_right | spatial2_discrete | 0.04737349 | 0.04294265 | 0.10256484 | 14.26785941 | 0.19966838 |
| left_right | global_ttt | 0.05308801 | 0.04708993 | 0.09763383 | 13.06183516 | 0.06583794 |
| left_right | spatial2_ttt | 0.04284468 | 0.03464573 | 0.09438481 | 14.52406983 | 0.24091861 |
| quadrants | identity | 0.05984607 | 0.05988720 | 0.10190258 | 12.49338977 | 0.00000000 |
| quadrants | global_direct | 0.06044747 | 0.06253698 | 0.09532195 | 12.42953555 | -0.01962904 |
| quadrants | spatial2_direct | 0.04680828 | 0.04492547 | 0.07279853 | 13.54183580 | 0.20432371 |
| quadrants | global_discrete | 0.06212251 | 0.05984416 | 0.11138455 | 12.42484396 | -0.14010191 |
| quadrants | spatial2_discrete | 0.06999890 | 0.06624125 | 0.12429080 | 11.94112985 | -0.28957126 |
| quadrants | global_ttt | 0.05205186 | 0.04773208 | 0.09400885 | 13.11633399 | 0.07239640 |
| quadrants | spatial2_ttt | 0.05761221 | 0.05445689 | 0.09688736 | 12.76437583 | -0.01579214 |
| smooth_gradient | identity | 0.02042468 | 0.01872252 | 0.03352607 | 17.27119335 | 0.00000000 |
| smooth_gradient | global_direct | 0.02147585 | 0.01860365 | 0.03645195 | 17.08680841 | -0.05787013 |
| smooth_gradient | spatial2_direct | 0.01981833 | 0.01699436 | 0.03352607 | 17.39928046 | 0.01456048 |
| smooth_gradient | global_discrete | 0.03048533 | 0.02429289 | 0.07093719 | 15.97380706 | -0.52898191 |
| smooth_gradient | spatial2_discrete | 0.03586771 | 0.02672245 | 0.09885374 | 15.50471968 | -0.77150330 |
| smooth_gradient | global_ttt | 0.02525005 | 0.02125637 | 0.05739232 | 16.64031004 | -0.31305367 |
| smooth_gradient | spatial2_ttt | 0.02690429 | 0.01979288 | 0.05886439 | 16.63010707 | -0.34131439 |
| heterogeneous | identity | 0.05992154 | 0.05845933 | 0.10190258 | 12.50577171 | 0.00000000 |
| heterogeneous | global_direct | 0.06044955 | 0.06017987 | 0.09553175 | 12.45303240 | -0.01607919 |
| heterogeneous | spatial2_direct | 0.04377792 | 0.04094116 | 0.06942726 | 13.88545765 | 0.25523732 |
| heterogeneous | global_discrete | 0.06367791 | 0.06014208 | 0.11816922 | 12.32680495 | -0.14077884 |
| heterogeneous | spatial2_discrete | 0.05868620 | 0.05470663 | 0.10844379 | 13.10449463 | -0.04495144 |
| heterogeneous | global_ttt | 0.05256994 | 0.04761957 | 0.09502922 | 13.08908457 | 0.06911717 |
| heterogeneous | spatial2_ttt | 0.05022844 | 0.04515640 | 0.09617316 | 13.64422283 | 0.11256324 |

Perfect reconstruction PSNR is +infinity (null in JSON), with perfect counts saved separately. Smooth gradient is report-only. All per-image pairwise MSE differences are saved, positive A-minus-B means B is better.

# T011 projected spatial TTT

Qualified: False; failed: ['clean_p95', 'beyond_discrete']

| Criterion | Value | Pass |
|---|---:|---|
| clean_mean | 0.000962722 | True |
| clean_p95 | 0.005844413 | False |
| homogeneous_dark | 0.503560966 | True |
| homogeneous_bright | 0.452760397 | True |
| spatial_value | 0.539975121 | True |
| beyond_direct | 0.873594537 | True |
| beyond_discrete | 0.955410612 | False |
| dark_region_safety | 0.469980515 | True |
| bright_region_safety | 0.409855901 | True |
| renderer_support | 0.677177449 | True |

| Condition | Method | Mean MSE | P95 MSE | Mean PSNR (finite) | Mean updates |
|---|---|---:|---:|---:|---:|
| clean | identity | 0.00000000 | 0.00000000 | null | 0.00000000 |
| clean | region2_direct | 0.00178163 | 0.00943066 | 23.45144181 | 0.00000000 |
| clean | region2_discrete_projected | 0.00091962 | 0.00453982 | 25.70048371 | 0.00000000 |
| clean | global_ttt_envelope | 0.00264919 | 0.01587692 | 21.27175608 | 8.45000000 |
| clean | region2_ttt_envelope | 0.00129178 | 0.00660740 | 24.03910476 | 8.45000000 |
| clean | global_ttt_projected | 0.00152179 | 0.01326566 | 23.91459623 | 8.70000000 |
| clean | bilinear2_ttt_projected | 0.00084000 | 0.00408080 | 27.10403951 | 9.00000000 |
| clean | region2_ttt_projected_1step | 0.00003394 | 0.00017186 | 40.86050133 | 0.22500000 |
| clean | region2_ttt_projected | 0.00096272 | 0.00584441 | 27.27291569 | 8.70000000 |
| homogeneous_dark | identity | 0.07531356 | 0.14799467 | 11.68360522 | 0.00000000 |
| homogeneous_dark | region2_direct | 0.04664501 | 0.08613750 | 13.90467029 | 0.00000000 |
| homogeneous_dark | region2_discrete_projected | 0.04022667 | 0.08494204 | 15.08000525 | 0.00000000 |
| homogeneous_dark | global_ttt_envelope | 0.01962023 | 0.08494204 | 19.54973350 | 17.20000000 |
| homogeneous_dark | region2_ttt_envelope | 0.03315989 | 0.08494204 | 16.46530703 | 16.60000000 |
| homogeneous_dark | global_ttt_projected | 0.02579301 | 0.08494204 | 17.41017869 | 23.12500000 |
| homogeneous_dark | bilinear2_ttt_projected | 0.03730391 | 0.08494204 | 15.42635985 | 26.87500000 |
| homogeneous_dark | region2_ttt_projected_1step | 0.06992190 | 0.13329407 | 12.01085654 | 0.90000000 |
| homogeneous_dark | region2_ttt_projected | 0.03792497 | 0.08494204 | 15.36241710 | 22.97500000 |
| homogeneous_bright | identity | 0.03850826 | 0.05756059 | 14.52754894 | 0.00000000 |
| homogeneous_bright | region2_direct | 0.01424613 | 0.03403071 | 19.70815437 | 0.00000000 |
| homogeneous_bright | region2_discrete_projected | 0.01783745 | 0.03407822 | 18.01030737 | 0.00000000 |
| homogeneous_bright | global_ttt_envelope | 0.01296026 | 0.02719799 | 19.78045864 | 29.30000000 |
| homogeneous_bright | region2_ttt_envelope | 0.01970950 | 0.03819959 | 17.75900232 | 29.15000000 |
| homogeneous_bright | global_ttt_projected | 0.01095290 | 0.02370354 | 20.47586691 | 29.30000000 |
| homogeneous_bright | bilinear2_ttt_projected | 0.01577040 | 0.03611195 | 18.70218135 | 29.85000000 |
| homogeneous_bright | region2_ttt_projected_1step | 0.03183951 | 0.04586220 | 15.32299505 | 0.97500000 |
| homogeneous_bright | region2_ttt_projected | 0.01743502 | 0.03819751 | 18.24732660 | 29.15000000 |
| left_right | identity | 0.05694992 | 0.08470147 | 12.80369121 | 0.00000000 |
| left_right | region2_direct | 0.02932787 | 0.04975938 | 15.83249049 | 0.00000000 |
| left_right | region2_discrete_projected | 0.02737811 | 0.04827434 | 16.28583403 | 0.00000000 |
| left_right | global_ttt_envelope | 0.04575066 | 0.06696544 | 13.68647878 | 33.90000000 |
| left_right | region2_ttt_envelope | 0.02290434 | 0.04777683 | 17.39327351 | 25.92500000 |
| left_right | global_ttt_projected | 0.04732841 | 0.07642413 | 13.52309420 | 34.45000000 |
| left_right | bilinear2_ttt_projected | 0.03379049 | 0.05810014 | 15.24869088 | 35.30000000 |
| left_right | region2_ttt_projected_1step | 0.05074280 | 0.07550784 | 13.30508215 | 1.00000000 |
| left_right | region2_ttt_projected | 0.02590639 | 0.04777684 | 16.61457285 | 29.07500000 |
| quadrants | identity | 0.05686745 | 0.09001451 | 12.88272684 | 0.00000000 |
| quadrants | region2_direct | 0.02926011 | 0.05047156 | 15.87384324 | 0.00000000 |
| quadrants | region2_discrete_projected | 0.02619273 | 0.05197927 | 16.55483330 | 0.00000000 |
| quadrants | global_ttt_envelope | 0.04505821 | 0.08321491 | 13.79972268 | 36.40000000 |
| quadrants | region2_ttt_envelope | 0.02251598 | 0.04772671 | 17.55071056 | 25.92500000 |
| quadrants | global_ttt_projected | 0.04745771 | 0.08573519 | 13.58159926 | 36.40000000 |
| quadrants | bilinear2_ttt_projected | 0.03732516 | 0.06028397 | 14.73750816 | 39.42500000 |
| quadrants | region2_ttt_projected_1step | 0.05060519 | 0.07689967 | 13.39789526 | 1.00000000 |
| quadrants | region2_ttt_projected | 0.02527575 | 0.05222778 | 16.82521357 | 28.22500000 |
| offset_left_right_40 | identity | 0.05417787 | 0.07932888 | 13.01070067 | 0.00000000 |
| offset_left_right_40 | region2_direct | 0.03873693 | 0.06429772 | 14.54182906 | 0.00000000 |
| offset_left_right_40 | region2_discrete_projected | 0.03837711 | 0.06252634 | 14.55942371 | 0.00000000 |
| offset_left_right_40 | global_ttt_envelope | 0.05094037 | 0.11903862 | 13.34900970 | 31.20000000 |
| offset_left_right_40 | region2_ttt_envelope | 0.03918812 | 0.06242669 | 14.47207406 | 29.97500000 |
| offset_left_right_40 | global_ttt_projected | 0.04792345 | 0.07886553 | 13.54429368 | 31.25000000 |
| offset_left_right_40 | bilinear2_ttt_projected | 0.03882676 | 0.07086723 | 14.57462314 | 32.77500000 |
| offset_left_right_40 | region2_ttt_projected_1step | 0.05006790 | 0.07217074 | 13.35556233 | 0.95000000 |
| offset_left_right_40 | region2_ttt_projected | 0.03933259 | 0.06789079 | 14.49199997 | 31.22500000 |
| heterogeneous | identity | 0.05690868 | 0.09001451 | 12.84320902 | 0.00000000 |
| heterogeneous | region2_direct | 0.02929399 | 0.05010457 | 15.85316687 | 0.00000000 |
| heterogeneous | region2_discrete_projected | 0.02678542 | 0.05197927 | 16.42033367 | 0.00000000 |
| heterogeneous | global_ttt_envelope | 0.04540443 | 0.07640242 | 13.74310073 | 35.15000000 |
| heterogeneous | region2_ttt_envelope | 0.02271016 | 0.04772671 | 17.47199204 | 25.92500000 |
| heterogeneous | global_ttt_projected | 0.04739306 | 0.08354854 | 13.55234673 | 35.42500000 |
| heterogeneous | bilinear2_ttt_projected | 0.03555782 | 0.05976271 | 14.99309952 | 37.36250000 |
| heterogeneous | region2_ttt_projected_1step | 0.05067400 | 0.07689967 | 13.35148870 | 1.00000000 |
| heterogeneous | region2_ttt_projected | 0.02559107 | 0.05222778 | 16.71989321 | 28.65000000 |

One-step is diagnostic only; offset_left_right_40 is report-only and excluded from qualification.
Perfect PSNR is infinite (null in JSON); no automatic method promotion, tuning, or future task.

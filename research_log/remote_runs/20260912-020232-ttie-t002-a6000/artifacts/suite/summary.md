# T002 complete matrix summary

504 rows; all finite: True. Means across all declared seeds. No per-case selection.

Max |uniform96-global| MSE: 9.84582584e-06; max |spatial1-global| MSE: 0.

Uniform96 still renders a six-dimensional global function; extra symmetric latent scalars do not add spatial or nonlinear expressivity.

## midtone — evaluation MSE

| Condition | identity | global | uniform96 | spatial1 | spatial2 | spatial4 | spatial8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0000000 | 0.0005575 | 0.0005491 | 0.0005575 | 0.0005576 | 0.0005393 | 0.0004748 |
| homogeneous_dark | 0.0775870 | 0.0011337 | 0.0011337 | 0.0011337 | 0.0011337 | 0.0011337 | 0.0011339 |
| homogeneous_bright | 0.0773811 | 0.0003329 | 0.0003329 | 0.0003329 | 0.0003330 | 0.0003331 | 0.0003335 |
| left_right | 0.0774843 | 0.0098803 | 0.0098803 | 0.0098803 | 0.0042488 | 0.0024584 | 0.0015818 |
| quadrants | 0.0774823 | 0.0098743 | 0.0098743 | 0.0098743 | 0.0056981 | 0.0040560 | 0.0038780 |
| smooth_gradient | 0.0263874 | 0.0058404 | 0.0058404 | 0.0058404 | 0.0022995 | 0.0010770 | 0.0007939 |
| stripes_4 | 0.0775185 | 0.0122895 | 0.0122895 | 0.0122895 | 0.0121535 | 0.0120531 | 0.0115963 |
| stripes_12 | 0.0775870 | 0.0504757 | 0.0504809 | 0.0504757 | 0.0513388 | 0.0520596 | 0.0535242 |

## dark_structures — evaluation MSE

| Condition | identity | global | uniform96 | spatial1 | spatial2 | spatial4 | spatial8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0000000 | 0.0176932 | 0.0176932 | 0.0176932 | 0.0179353 | 0.0203560 | 0.0262865 |
| homogeneous_dark | 0.0583337 | 0.0153570 | 0.0153570 | 0.0153570 | 0.0156031 | 0.0183375 | 0.0263029 |
| homogeneous_bright | 0.0581796 | 0.0171524 | 0.0171524 | 0.0171524 | 0.0173637 | 0.0193812 | 0.0256809 |
| left_right | 0.0582503 | 0.0188019 | 0.0188019 | 0.0188019 | 0.0172578 | 0.0202654 | 0.0273744 |
| quadrants | 0.0582557 | 0.0214037 | 0.0214037 | 0.0214037 | 0.0189669 | 0.0204942 | 0.0283540 |
| smooth_gradient | 0.0212078 | 0.0183327 | 0.0183327 | 0.0183327 | 0.0181473 | 0.0198688 | 0.0263475 |
| stripes_4 | 0.0582820 | 0.0242696 | 0.0242696 | 0.0242696 | 0.0250393 | 0.0273358 | 0.0344770 |
| stripes_12 | 0.0583337 | 0.0311336 | 0.0311336 | 0.0311336 | 0.0313740 | 0.0367338 | 0.0572289 |

## high_key — evaluation MSE

| Condition | identity | global | uniform96 | spatial1 | spatial2 | spatial4 | spatial8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0000000 | 0.0628824 | 0.0628824 | 0.0628824 | 0.0628824 | 0.0628824 | 0.0628825 |
| homogeneous_dark | 0.1706502 | 0.0630230 | 0.0630230 | 0.0630230 | 0.0630230 | 0.0630230 | 0.0630231 |
| homogeneous_bright | 0.0641102 | 0.0641238 | 0.0641238 | 0.0641238 | 0.0641238 | 0.0641238 | 0.0641237 |
| left_right | 0.1173850 | 0.0787740 | 0.0787741 | 0.0787740 | 0.0667231 | 0.0649478 | 0.0641582 |
| quadrants | 0.1173728 | 0.0787712 | 0.0787713 | 0.0787712 | 0.0688427 | 0.0661748 | 0.0657750 |
| smooth_gradient | 0.0478751 | 0.0697532 | 0.0697532 | 0.0697532 | 0.0641582 | 0.0634104 | 0.0632007 |
| stripes_4 | 0.1221678 | 0.0770667 | 0.0770667 | 0.0770667 | 0.0768556 | 0.0767123 | 0.0721431 |
| stripes_12 | 0.1317571 | 0.1081923 | 0.1081926 | 0.1081923 | 0.1081576 | 0.1081256 | 0.1080729 |

## Spatial4 compared with uniform96

| Family | Condition | MSE reduction % | Input clipping % | Spatial4 output clipping % |
| --- | --- | ---: | ---: | ---: |
| midtone | clean | 1.798 | 0.000 | 0.000 |
| midtone | homogeneous_dark | -0.002 | 0.000 | 0.000 |
| midtone | homogeneous_bright | -0.066 | 1.698 | 0.000 |
| midtone | left_right | 75.118 | 0.864 | 0.000 |
| midtone | quadrants | 58.924 | 0.911 | 0.000 |
| midtone | smooth_gradient | 81.560 | 0.022 | 0.000 |
| midtone | stripes_4 | 1.923 | 0.555 | 0.000 |
| midtone | stripes_12 | -3.127 | 0.000 | 0.000 |
| dark_structures | clean | -15.050 | 0.000 | 0.000 |
| dark_structures | homogeneous_dark | -19.408 | 0.000 | 0.000 |
| dark_structures | homogeneous_bright | -12.994 | 1.270 | 0.000 |
| dark_structures | left_right | -7.784 | 0.702 | 0.000 |
| dark_structures | quadrants | 4.249 | 0.684 | 0.000 |
| dark_structures | smooth_gradient | -8.379 | 0.022 | 0.000 |
| dark_structures | stripes_4 | -12.634 | 0.418 | 0.000 |
| dark_structures | stripes_12 | -17.988 | 0.000 | 0.000 |
| high_key | clean | -0.000 | 0.000 | 0.000 |
| high_key | homogeneous_dark | -0.000 | 0.000 | 0.000 |
| high_key | homogeneous_bright | -0.000 | 100.000 | 0.000 |
| high_key | left_right | 17.552 | 50.000 | 0.000 |
| high_key | quadrants | 15.991 | 50.000 | 0.000 |
| high_key | smooth_gradient | 9.093 | 19.511 | 0.000 |
| high_key | stripes_4 | 0.460 | 50.000 | 0.000 |
| high_key | stripes_12 | 0.062 | 50.000 | 0.000 |

Clean-condition MSE is identity drift. Identity PSNR for zero MSE is infinite (null in JSON, blank in CSV). Non-clean input_output_drift_mse is the magnitude of editing, not reference recovery error.

High-frequency stripe results also reflect 8x8 patch-loss averaging/aliasing, content priors, operator flexibility and clipping; they do not isolate representation resolution alone.

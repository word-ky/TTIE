# T051-A fixed spatial exposure probe

Base1160d117, accepted T050. Only zero8x8u trainable;2tanh -> bilinear align_corners=False -> multiply clamped affine by2**e -> clamp -> frozen tone -> accepted hard gate. Adam.05,500updates,one start,GPU1. All100 low-only preflight before normals; freeze then metrics/replay. Gate pairedmeanPSNR>=1.5,median>=.75,meanSSIM>=0. No sweeps/test/deployablechanges. GitHub emailverification403 currently blocks publication; preserve commits locally.

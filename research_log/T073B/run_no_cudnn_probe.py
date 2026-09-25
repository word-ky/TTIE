"""Probe native-size synthetic inference with cuDNN convolution disabled."""

import runpy
import sys

import torch

torch.backends.cudnn.enabled = False
sys.argv[0] = "run_low_only.py"
runpy.run_path(__file__.replace("run_no_cudnn_probe.py", "run_low_only.py"), run_name="__main__")

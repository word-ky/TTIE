#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -u -m ttie.nonspatial_target --output "$AUTODL_ARTIFACTS_DIR/audit" --source-sha "$1" --device cuda:0

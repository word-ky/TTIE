#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -u -m ttie.nonspatial_oof extract --output "$AUTODL_ARTIFACTS_DIR/features" --source-sha "$1" --device cuda:0

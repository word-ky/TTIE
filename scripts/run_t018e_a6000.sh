#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source_sha="$1"
images=/home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017
cohort="$AUTODL_ARTIFACTS_DIR/cohort"
selected="$AUTODL_ARTIFACTS_DIR/selected"
python -u -m ttie.fresh_direction.prepare manifest --images "$images" --output "$cohort" --source-sha "$source_sha"
python -u -m ttie.fresh_direction.prepare synthesize --images "$images" --output "$cohort" --source-sha "$source_sha" --device cuda:0
python -u -m ttie.fresh_direction.select --cohort "$cohort" --output "$selected" --source-sha "$source_sha" --device cuda:0
echo "GPU features complete. Local immutable CPU inference must freeze all decisions before separate evaluation."

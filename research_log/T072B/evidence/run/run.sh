#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-195909-ttie-t072b-uhdll-native'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-195909-ttie-t072b-uhdll-native'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-195909-ttie-t072b-uhdll-native/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t072b-uhdll-native && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONPATH=$PWD TTIE_SOURCE_COMMIT=4cd6ab605c0ba8648f78b651ac2d97a269044f90 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T072B.run --method ours --low /media/wenchang/F/wjq/TTIE/shared/t072b/1003_UHD_LL.JPG --out /media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native/ours && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T072B.run --method retinexformer --low /media/wenchang/F/wjq/TTIE/shared/t072b/1003_UHD_LL.JPG --out /media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native/retinexformer && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T072B.run --method snr_aware --low /media/wenchang/F/wjq/TTIE/shared/t072b/1003_UHD_LL.JPG --out /media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native/snr_aware
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status

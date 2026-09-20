#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-001832-ttie-t066c-reentry'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-001832-ttie-t066c-reentry'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-001832-ttie-t066c-reentry/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260921-ttie-t066c-reentry && export PYTHONDONTWRITEBYTECODE=1 TTIE_SOURCE_COMMIT=4e4c663827ff0bc7d7b87364c57fd2407ce0bc4d && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T066C/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066C.run --out /media/wenchang/F/wjq/TTIE/runs/T066C-prefix-reentry && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066C.verify --out /media/wenchang/F/wjq/TTIE/runs/T066C-prefix-reentry
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status

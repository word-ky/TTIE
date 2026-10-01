#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
round1_log=/media/wenchang/F/wjq/TTIE/runs/20260929-162005-ttie-smid-ceiling-r1-gpu0/train.log

while ! grep -q '^\[autodl\] exit_code=' "$round1_log"; do
  sleep 30
done
grep -q '^\[autodl\] exit_code=0$' "$round1_log"

printf '[ceiling-batch] SMID round 1 succeeded at %s\n' "$(date -Is)"
bash "$root/run_smid_ceiling_finish.sh"
printf '[ceiling-batch] SMID ceiling complete at %s\n' "$(date -Is)"

bash "$root/run_sid_ceiling_round1.sh"
printf '[ceiling-batch] SID round 1 complete at %s\n' "$(date -Is)"
bash "$root/run_sid_ceiling_finish.sh"
printf '[ceiling-batch] SID ceiling complete at %s\n' "$(date -Is)"

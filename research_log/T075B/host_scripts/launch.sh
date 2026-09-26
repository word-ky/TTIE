#!/bin/bash
# usage: launch.sh NAME CWD cmd...   (detached; GPU-serialized via flock; log + rc under T075B/logs)
N=$1; D=$2; shift 2
L=/root/autodl-tmp/TTIE/T075B/logs
test -e $L/$N.rc && { echo "rc exists for $N"; exit 1; }
cd "$D" || exit 1
setsid nohup bash -c "flock /root/autodl-tmp/TTIE/gpu.lock \"\$@\"; echo \$? > $L/$N.rc" _ "$@" > $L/$N.log 2>&1 < /dev/null &
echo launched $N pid $!

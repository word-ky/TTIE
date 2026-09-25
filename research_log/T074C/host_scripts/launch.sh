#!/bin/bash
# usage: launch.sh NAME CWD cmd...   (detached; log + rc file under T074C/logs)
N=$1; D=$2; shift 2
L=/root/autodl-tmp/TTIE/T074C/logs
test -e $L/$N.rc && { echo "rc exists for $N"; exit 1; }
cd "$D" || exit 1
setsid nohup bash -c "\"\$@\"; echo \$? > $L/$N.rc" _ "$@" > $L/$N.log 2>&1 < /dev/null &
echo launched $N pid $!

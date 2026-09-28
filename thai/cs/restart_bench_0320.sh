#!/usr/bin/env bash
# Stop a running bench_0320.sh (and its containers), update the ~/laya-0320 clone, start it again.
pkill -f 'bash thai/cs/bench_0320.sh' || true
for c in $(docker ps -q --filter ancestor=laya-train-0320); do docker rm -f "$c"; done
cd ~/laya-0320 || exit 1
rm -f thai/cs/diag_tilelang.sh thai/cs/diag_tilelang2.sh
git pull -q && git log --oneline -1
sed -i 's/\r$//' thai/cs/*.sh
(nohup bash thai/cs/bench_0320.sh > ~/laya/thai/out/bench_0320.log 2>&1 < /dev/null &)
sleep 3
cat ~/laya/thai/out/bench_0320.log

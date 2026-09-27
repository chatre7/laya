#!/usr/bin/env bash
# After run8.sh finishes: score every checkpoint on the krathu-500 sentiment set (real Pantip comments), GPU 1.
#   nohup bash thai/cs/post_run8.sh > thai/out/post_run8.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
echo "== [$(date +%H:%M)] waiting for run8.sh"
while ! grep -q "== \[.*\] done" thai/out/run8.log 2>/dev/null; do sleep 120; done
echo "== [$(date +%H:%M)] run 8 done; krathu-500 sentiment eval"
for r in run3 run5 run6 run7 run8; do
  [ -d thai/out/laya-th-$r ] || continue
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/domain/krathu500_eval.jsonl --out /work/thai/out/${r}_krathu.json
done
"${RUN[@]}" chmod -R a+rX /work/thai/out
echo "== [$(date +%H:%M)] done"

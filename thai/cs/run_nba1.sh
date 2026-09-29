#!/usr/bin/env bash
# Next best action, run nba1 (banking), unattended: teacher labels -> items (+ replay) -> train from run 8 -> eval (NBA on the
# 120 real rows x 4 contexts; the real Pantip intent set for regressions) -> restore :8011.
#   nohup bash thai/cs/run_nba1.sh > thai/out/nba1.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out thai/data/nba

echo "== [$(date +%H:%M)] 1/5 teacher labels (message only; the playbook spreads it over 4 contexts) + replay items"
"${CPU[@]}" python label_nba.py --teacher http://172.18.72.145:8010 --workers 8

echo "== [$(date +%H:%M)] 2/5 baselines before training: run 8 asked directly, teacher, rule table"
docker stop laya-cascade >/dev/null 2>&1 || true
"${RUN[@]}" python eval_nba.py --models /work/thai/out/laya-th-run8 --rule-model /work/thai/out/laya-th-run8 \
  --teacher http://172.18.72.145:8010 --out /work/thai/out/nba_baselines.json

echo "== [$(date +%H:%M)] 3/5 train nba1 from run 8, 2 epochs (cascade stopped)"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run8 --items /work/thai/data/nba/nba1_items.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-nba1

echo "== [$(date +%H:%M)] 4/5 eval nba1: NBA, and the real Pantip intent set (run 8 = 0.656)"
"${RUN[@]}" python eval_nba.py --models /work/thai/out/laya-th-nba1 --rule-model /work/thai/out/laya-th-nba1 --teacher "" \
  --out /work/thai/out/nba1_eval.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-nba1 --eval /work/thai/data/domain/real_cs_eval.jsonl \
  --out /work/thai/out/nba1_real.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (call center, run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"

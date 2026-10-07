#!/usr/bin/env bash
# Parity of run 19 under the fork's code (~/laya, 0.3.20) and the upstream-merged branch (~/laya-sync, 0.3.28): the same
# checkpoint and 200 texts, answers compared. The sync tree borrows the model and data folders of ~/laya read-only.
#   bash thai/cs/parity.sh
set -uo pipefail
cd ~
[ -d ~/laya-sync/.git ] && (cd ~/laya-sync && git fetch -q && git checkout -q sync-upstream && git pull -q) \
  || git clone -q --branch sync-upstream https://github.com/chatre7/laya.git ~/laya-sync || { echo "clone failed"; exit 1; }
echo "sync tree: $(cd ~/laya-sync && git log --oneline -1)"
G='"device=1"'
docker run --rm --gpus "$G" -v ~/laya:/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train \
  python parity_check.py --out /work/thai/out/parity_old.json 2>&1 | grep -v -iE "warn" | tail -2
docker run --rm --gpus "$G" -v ~/laya-sync:/work -v ~/laya/thai/out:/work/thai/out -v ~/laya/thai/data_domain:/work/thai/data_domain:ro \
  -w /work/thai/cs -v docker_hf-cache:/hf laya-train python parity_check.py --out /work/thai/out/parity_new.json 2>&1 | grep -v -iE "warn" | tail -12
docker run --rm -v ~/laya:/work -w /work/thai/cs laya-train python parity_check.py --compare /work/thai/out/parity_old.json /work/thai/out/parity_new.json

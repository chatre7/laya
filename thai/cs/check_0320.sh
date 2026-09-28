#!/usr/bin/env bash
# Does laya 0.3.20 (branch thai-0.3.20) reproduce the run 8 numbers? Runs in a separate clone (~/laya-0320) with the existing
# data / checkpoints mounted in, so ~/laya (0.3.5, serving :8011) is untouched. GPU 1 is shared with the cascade (3 GB used).
#   nohup bash ~/laya-0320/thai/cs/check_0320.sh > ~/laya/thai/out/check_0320.log 2>&1 &
set -euo pipefail
cd ~/laya-0320
DATA=~/laya/thai/data
OUT=~/laya/thai/out
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -v "$DATA":/work/thai/data:ro -v "$OUT":/work/thai/out -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work laya-train)

echo "== [$(date +%H:%M)] laya version in the clone"
"${CPU[@]}" python -c "import laya, sys; print('laya', laya.__version__, 'python', sys.version.split()[0])"

echo "== [$(date +%H:%M)] 1/3 unit tests (CPU, test_batch.py is a script not a pytest module)"
"${CPU[@]}" python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_batch.py --ignore=tests/test_ts_parity.py -x 2>&1 | tail -15 || echo "TESTS FAILED (see above)"

echo "== [$(date +%H:%M)] 2/3 run 8 on the real Pantip set with 0.3.20"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run8 --eval /work/thai/data/domain/real_cs_eval.jsonl --out /work/thai/out/run8_realcs8_0320.json

echo "== [$(date +%H:%M)] 3/3 run 8 on the public set with 0.3.20"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run8 --eval /work/thai/data/eval.jsonl --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run8_0320.json

"${RUN[@]}" chmod -R a+rX /work/thai/out || true
echo "== [$(date +%H:%M)] done"

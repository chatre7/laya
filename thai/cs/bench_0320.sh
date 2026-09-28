#!/usr/bin/env bash
# Build the 0.3.20 image (TileLang + ONNX extras) from the ~/laya-0320 clone and benchmark the fast path on run 8, GPU 1
# (shared with the :8011 cascade). Then the upstream unit tests on CPU.
#   nohup bash ~/laya-0320/thai/cs/bench_0320.sh > ~/laya/thai/out/bench_0320.log 2>&1 &
set -euo pipefail
cd ~/laya-0320
git fetch -q && git checkout -q thai && git pull -q && git log --oneline -1
DATA=~/laya/thai/data
OUT=~/laya/thai/out
echo "== [$(date +%H:%M)] build laya-train-0320"
docker build -q -t laya-train-0320 -f thai/Dockerfile.0320 . | tail -1
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -v "$DATA":/work/thai/data:ro -v "$OUT":/work/thai/out -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train-0320)
"${RUN[@]}" python -c "import laya, tilelang, onnxruntime; print('laya', laya.__version__, 'tilelang', tilelang.__version__, 'ort', onnxruntime.__version__)"

echo "== [$(date +%H:%M)] 1/2 fast-path benchmark on run 8 (real Pantip states, 3 questions)"
"${RUN[@]}" python bench_fast.py --model /work/thai/out/laya-th-run8 --out /work/thai/out/bench_fast8.json 2>&1 | grep -v "Warning\|it/s\]"

echo "== [$(date +%H:%M)] 2/2 upstream unit tests (CPU)"
docker run --rm -v "$PWD":/work -w /work laya-train-0320 python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_batch.py --ignore=tests/test_ts_parity.py 2>&1 | tail -8 || echo "TESTS FAILED"
"${RUN[@]}" chmod -R a+rX /work/thai/out || true
echo "== [$(date +%H:%M)] done"

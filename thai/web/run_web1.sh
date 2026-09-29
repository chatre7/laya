#!/usr/bin/env bash
# Web-agent domain, run web1, unattended: Mind2Web text -> element-choice records -> Thai task translations (Qwen3-4B on GPU 1)
# -> items -> train laya from run 3 (general Thai distillation, not the call-center model) -> eval on the three Mind2Web test
# splits in Thai and English. :8011 (call center, run 8) is stopped only while GPU 1 trains.
#   nohup bash thai/web/run_web1.sh > thai/out/web1.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/web -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/web -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out thai/data/web

echo "== [$(date +%H:%M)] 1/6 Mind2Web text columns -> records"
"${CPU[@]}" python prep_m2w.py --out /work/thai/data/web

echo "== [$(date +%H:%M)] 2/6 vLLM Qwen3-4B on GPU 1 (:8012), translate tasks"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker run -d --name vllm-rewrite --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B --served-model-name qwen3-4b --dtype bfloat16 --gpu-memory-utilization 0.62 --max-model-len 2048 --max-num-seqs 64 >/dev/null
for i in $(seq 1 120); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b && break; sleep 5; done
curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b || { echo "vLLM did not come up"; docker logs vllm-rewrite | tail -20; exit 1; }
python3 thai/web/translate_tasks.py --data thai/data/web --out thai/data/web/tasks_th.json
docker rm -f vllm-rewrite >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 3/6 items + eval files"
"${CPU[@]}" python label_web.py --data /work/thai/data/web --student /work/thai/out/laya-th-run3

echo "== [$(date +%H:%M)] 4/6 train web1 from run 3, 3 epochs (cascade stopped)"
docker stop laya-cascade >/dev/null 2>&1 || true
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/web/web1_items.pt \
  --head-max-len 768 --epochs 3 --out /work/thai/out/laya-th-web1

echo "== [$(date +%H:%M)] 5/6 eval: Mind2Web test splits, Thai and English (web1, and run 3 as the untrained baseline)"
for m in web1 run3; do
  for s in test_task test_website test_domain; do
    for l in th en; do
      "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$m --eval /work/thai/data/web/web_eval_${s}_${l}.jsonl \
        --out /work/thai/out/${m}_m2w_${s}_${l}.json
    done
  done
done

echo "== [$(date +%H:%M)] 6/6 restore :8011 (call center, run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"

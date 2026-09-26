#!/usr/bin/env bash
# Real-question English sets -> Thai colloquial (run 8 inputs): banking77 (2 variants) and insurance-qa (relevant topics, 1 variant).
# vLLM runs on GPU 0 next to the teacher (GPU 1 is training run 7). No teacher gate here: the banking77 intents are not in
# cs_questions yet; the gate runs after the taxonomy is decided.
#   nohup bash thai/cs/prep_cs2.sh > thai/data/cs2/prep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p thai/data/cs2

echo "== [$(date +%H:%M)] 1/3 vLLM qwen3-4b on GPU 0 (:8012)"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker run -d --name vllm-rewrite --gpus '"device=0"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B --served-model-name qwen3-4b --dtype bfloat16 --gpu-memory-utilization 0.60 --max-model-len 4096 --max-num-seqs 64 >/dev/null
for i in $(seq 1 120); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b && break; sleep 5; done
curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b || { echo "vLLM did not come up"; docker logs vllm-rewrite | tail -20; exit 1; }

echo "== [$(date +%H:%M)] 2/3 insurance-qa: keep relevant topics"
python3 - <<'PY'
import json
keep = {"life-insurance", "auto-insurance", "health-insurance", "home-insurance", "renters-insurance", "travel-insurance", "pet-insurance", "critical-illness-insurance", "other-insurance"}
rows = [json.loads(l) for l in open("thai/data/cs2/insuranceqa_en.jsonl", encoding="utf-8")]
sel = [r for r in rows if r["category"].lower() in keep]
with open("thai/data/cs2/insuranceqa_en_sel.jsonl", "w", encoding="utf-8") as f:
    for r in sel:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"insurance-qa kept {len(sel)}/{len(rows)}")
PY

echo "== [$(date +%H:%M)] 3/3 EN -> Thai colloquial: banking77 x2, insurance-qa x1"
python3 thai/cs/translate_colloquial.py --inp thai/data/cs2/banking77_en.jsonl --out thai/data/cs2/banking77_th.jsonl --variants 2 --workers 32 --seed 7
python3 thai/cs/translate_colloquial.py --inp thai/data/cs2/insuranceqa_en_sel.jsonl --out thai/data/cs2/insuranceqa_th.jsonl --variants 1 --workers 32 --seed 7
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
echo "== [$(date +%H:%M)] done"

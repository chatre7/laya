#!/usr/bin/env bash
# In-register out-of-scope (`other`) examples for run 7: vLLM on GPU 1, generate, teacher gate on the business question, stop vLLM.
#   nohup bash thai/cs/gen_other.sh > thai/data/cs/gen_other.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
N="${N:-6000}"
echo "== [$(date +%H:%M)] 1/3 vLLM qwen3-4b on GPU 1"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker run -d --name vllm-rewrite --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B --served-model-name qwen3-4b --dtype bfloat16 --gpu-memory-utilization 0.62 --max-model-len 4096 --max-num-seqs 64 >/dev/null
for i in $(seq 1 120); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b && break; sleep 5; done
curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b || { echo "vLLM did not come up"; exit 1; }

echo "== [$(date +%H:%M)] 2/3 generate $N out-of-scope messages"
python3 thai/cs/gen_other.py --out thai/data/cs/other_indomain.jsonl --n "$N" --workers 32

echo "== [$(date +%H:%M)] 3/3 teacher gate: keep where the teacher's business answer is other with p >= 0.4"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
python3 - <<'PY'
import json, sys, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "thai/cs")
from cs_questions import BUSINESS_Q
rows = [json.loads(l) for l in open("thai/data/cs/other_indomain.jsonl", encoding="utf-8")]
def ask(r):
    body = json.dumps({"state": r["text"], "questions": {"business": BUSINESS_Q}, "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request("http://localhost:8010/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["answers"]["business"]
        except Exception:
            pass
    return None
with ThreadPoolExecutor(8) as ex:
    ans = list(ex.map(ask, rows))
kept, by_kind = [], collections.defaultdict(lambda: [0, 0])
for r, a in zip(rows, ans):
    if a is None:
        continue
    p = a["probabilities"].get("other", 0.0)
    by_kind[r["kind"]][0] += 1
    if p >= 0.4:
        by_kind[r["kind"]][1] += 1
        kept.append({**r, "teacher_other_p": round(p, 3)})
with open("thai/data/cs/other_indomain_kept.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k, (n, m) in sorted(by_kind.items()):
    print(f"  {k:18s} {m:5d}/{n:<5d} kept ({m / max(1, n):.0%})")
print(f"kept {len(kept)}/{len(rows)}")
PY
echo "== [$(date +%H:%M)] done"

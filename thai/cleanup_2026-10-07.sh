#!/usr/bin/env bash
# Disk clean-up on the GPU box agreed on 2026-10-07 (groups A, B, D, E of the proposal): old laya run folders, trainer
# resume states, the Decision 2.0 / Clef downloads and image, and Docker leftovers that nothing uses. Keeps run 14 / 16 /
# 19, the serving images, the Qwen labellers, and everything that is not laya's (other projects' models and containers).
#   bash thai/cleanup_2026-10-07.sh
set -uo pipefail
cd ~/laya/thai
echo "before: $(df -h / | awk 'NR==2{print $4" free of "$2}')"

echo "== A. old run folders"
for r in 3 4 5 6 7 8 9 10 11 12 13 15 17 18; do [ -d "out/laya-th-run$r" ] && rm -rf "out/laya-th-run$r" && echo "  removed run $r"; done

echo "== B. trainer resume states in the kept runs"
for r in 14 16 19; do [ -d "out/laya-th-run$r/checkpoint_latest" ] && rm -rf "out/laya-th-run$r/checkpoint_latest" && echo "  removed run $r/checkpoint_latest"; done
ls -d out/laya-th-run* | tr '\n' ' '; echo

echo "== D. Decision 2.0 / Clef downloads (docker volume) and the clef image"
docker run --rm -v docker_hf-cache:/hf laya-train sh -c '
  for d in /hf/hub/models--Cloudflare--clef-flash /hf/hub/models--vllm-sr--Decision-2.0-Kai-0.6B /hf/hub/models--vllm-sr--Decision-2.0-Sol-2B /hf/hub/models--vllm-sr--Decision-2.0-Nox-4B; do
    [ -d "$d" ] && { du -sh "$d" | sed "s|/hf/hub/||"; rm -rf "$d"; }
  done
  rm -rf /hf/hub/.locks/models--Cloudflare--clef-flash /hf/hub/.locks/models--vllm-sr--Decision-2.0-* 2>/dev/null
  echo "  left in the volume:"; du -sh /hf/hub/* 2>/dev/null | sed "s|/hf/hub/|    |"'
docker rmi laya-train-clef >/dev/null 2>&1 && echo "  removed image laya-train-clef"

echo "== E. Docker leftovers"
echo "  stopped containers from our images:"
docker ps -a --filter status=exited --filter status=created --format '{{.ID}} {{.Image}} {{.Names}}' | grep -E ' (laya-train|laya-cascade|laya-train-clef|vllm/vllm-openai)' | tee /dev/stderr | awk '{print $1}' | xargs -r docker rm >/dev/null
echo "  other stopped containers (left alone):"
docker ps -a --filter status=exited --filter status=created --format '    {{.Image}} {{.Names}}'
for t in run3 run4 run5 run6 run7; do docker rmi "laya-cascade:$t" >/dev/null 2>&1 && echo "  removed image laya-cascade:$t"; done
docker image prune -f | tail -1
docker builder prune -f | tail -1

echo "after:  $(df -h / | awk 'NR==2{print $4" free of "$2}')"
docker ps --format '{{.Names}} {{.Status}}'

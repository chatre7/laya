#!/usr/bin/env bash
# Test desk for the call-center team at :8020 (no GPU: it calls the cascade at :8011). The laya-cascade image already has
# fastapi + uvicorn; the code is mounted read-only, feedback goes to thai/ui_data/feedback.jsonl on the host.
#   bash thai/ui/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ui_data && chmod 777 ui_data
docker rm -f laya-testui >/dev/null 2>&1 || true
docker run -d --name laya-testui --restart unless-stopped --no-healthcheck -p 8020:8020 \
  -e CASCADE_URL=http://172.18.72.145:8011 -e FEEDBACK_FILE=/data/feedback.jsonl \
  -v "$PWD":/app/thai:ro -v "$PWD/ui_data":/data -w /app/thai/ui \
  laya-cascade:run8 python -m uvicorn app:app --host 0.0.0.0 --port 8020
sleep 4
curl -s -m 5 http://localhost:8020/healthz && echo

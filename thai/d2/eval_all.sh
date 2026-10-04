#!/usr/bin/env bash
# Decision 2.0 models (vllm-sr), untrained, on our eval sets, beside the cascade on GPU 1 (nothing is stopped):
# new questions, hand-labelled reviews, real posts (long + short), mood against stars / ambiguous set (run 16 on the same
# sample), and the Thai suites of laya-thai-decisions. One model after another; a model that fails does not stop the rest.
#   nohup bash thai/d2/eval_all.sh vllm-sr/Decision-2.0-Kai-0.6B vllm-sr/Decision-2.0-Sol-2B > thai/out/d2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train)
QUIET='warn|Fetching|Downloading|Generating|it/s\]|B/s\]|^\[transformers\]|^- '
STARS_N=${STARS_N:-1000}
SUITES=synth_eval_domains_th,probe_noul_negation,probe_score_orientation,wisesight_sampler_eval,massive_intent_th
for M in "$@"; do
  N=$(basename "$M")
  echo "==== [$(date +%H:%M)] $N: new questions"
  "${RUN[@]}" python new_questions_demo.py --models "$M" 2>&1 | grep --line-buffered -v -iE "$QUIET"
  echo "==== [$(date +%H:%M)] $N: reviews (300 hand labels)"
  "${RUN[@]}" python eval_play.py --models "$M" --out "/work/thai/out/d2_${N}_play.json" 2>&1 | grep --line-buffered -v -iE "$QUIET"
  echo "==== [$(date +%H:%M)] $N: real posts"
  "${RUN[@]}" python eval_real.py --models "$M" --out "/work/thai/out/d2_${N}_real.json" 2>&1 | grep --line-buffered -v -iE "$QUIET"
  echo "==== [$(date +%H:%M)] $N: mood ($STARS_N reviews, ambiguous set, composed sets), run 16 on the same sample"
  "${RUN[@]}" python eval_stars.py --models "/work/thai/out/laya-th-run16,$M" --n "$STARS_N" --out "/work/thai/out/d2_${N}_stars.json" 2>&1 | grep --line-buffered -v -iE "$QUIET"
  echo "==== [$(date +%H:%M)] $N: laya-thai-decisions, Thai suites"
  "${RUN[@]}" python ../eval_decisions.py --models "$M" --suites "$SUITES" --out "/work/thai/out/d2_${N}_decisions.json" 2>&1 | grep --line-buffered -v -iE "$QUIET"
done
"${RUN[@]}" chmod -R a+rX /work/thai/out
echo "==== [$(date +%H:%M)] done"

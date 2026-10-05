"""Our evals read a score question as the rounded expected level. A model with flat probabilities then answers the middle
level every time. On the 300 hand-labelled reviews, urgency only: accuracy by rounded expectation and by the most probable
level, and which levels each reading picks.

    python urgency_check.py vllm-sr/Decision-2.0-Kai-0.6B /work/thai/out/laya-th-run16
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cs"))
from agents import load_agent  # noqa: E402
from cs_questions import SHARED  # noqa: E402

labels = json.load(open("/work/thai/data_domain/play_labels.json", encoding="utf-8"))["labels"]
rows = [json.loads(l) for l in open("/work/thai/data_domain/play_reviews_eval_sample.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["id"] in labels]
print("gold levels", dict(sorted(Counter(int(labels[r["id"]][2]) for r in rows).items())))
for path in sys.argv[1:]:
    agent = load_agent(path)
    rounded, top, t = Counter(), Counter(), time.perf_counter()
    ok_r = ok_t = 0
    for r in rows:
        a = agent.predict(r["text"], {"urgency": SHARED["urgency"]})["answers"]["urgency"]
        gold = int(labels[r["id"]][2])
        pr, pt = round(a["score"]), max(range(3), key=lambda i: a["probabilities"][str(i)])
        rounded[pr] += 1
        top[pt] += 1
        ok_r += pr == gold
        ok_t += pt == gold
    ms = (time.perf_counter() - t) * 1000 / len(rows)
    print(f"{os.path.basename(path):26s} rounded {ok_r / len(rows):.3f} {dict(sorted(rounded.items()))} | most probable {ok_t / len(rows):.3f} "
          f"{dict(sorted(top.items()))} | {ms:.0f} ms per one-question request", flush=True)
    del agent

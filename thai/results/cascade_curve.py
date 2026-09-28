"""Recompute the cascade curve from the per-decision rows a cascade.py run saved (no GPU), optionally per question.

    python cascade_curve.py cascade8_real.json
"""
import json
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [x for x in d["rows"] if x["t_ok"] is not None]
mean_s, mean_t = d["student_ms_per_record"], d["teacher_ms_per_record"]
print(f"{len(rows)} decisions; student only {sum(x['s_ok'] for x in rows) / len(rows):.3f}, teacher only {sum(x['t_ok'] for x in rows) / len(rows):.3f}")
by_q = defaultdict(list)
for x in rows:
    by_q[x["qid"]].append(x)
for qid, xs in [("all", rows)] + sorted(by_q.items()):
    print(f"== {qid} (n={len(xs)}): student {sum(x['s_ok'] for x in xs) / len(xs):.3f}  teacher {sum(x['t_ok'] for x in xs) / len(xs):.3f}")
    print(f"{'thr':>5s} {'acc':>6s} {'to_teacher':>10s} {'student@kept':>12s} {'teacher@sent':>12s} {'est ms':>7s}")
    for thr in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        sent = [x for x in xs if x["s_conf"] < thr]
        kept = [x for x in xs if x["s_conf"] >= thr]
        acc = (sum(x["s_ok"] for x in kept) + sum(x["t_ok"] for x in sent)) / len(xs)
        frac = len(sent) / len(xs)
        print(f"{thr:5.2f} {acc:6.3f} {frac:10.3f} {sum(x['s_ok'] for x in kept) / max(1, len(kept)):12.3f} "
              f"{sum(x['t_ok'] for x in sent) / max(1, len(sent)):12.3f} {mean_s + frac * mean_t:7.0f}")

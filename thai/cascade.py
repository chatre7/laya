"""Measure a student -> teacher cascade on the human-labelled eval set.

Every decision is answered by the laya student; if its confidence is below a threshold (or the question
has more than --max-options options) the OpenThai-SystemOne teacher answers instead. Reports accuracy,
teacher-call fraction and mean latency per threshold, so the operating point can be chosen from data.

    python cascade.py --student /work/thai/out/laya-th-distill --teacher http://172.18.72.145:8010 --out /work/thai/out/cascade.json
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import laya


def teacher_call(url, r):
    body = json.dumps({"state": r["state"], "questions": r["questions"], "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", body, {"content-type": "application/json"})
    for attempt in range(3):
        try:
            t = time.perf_counter()
            res = json.load(urllib.request.urlopen(req, timeout=120))
            return res["answers"], (time.perf_counter() - t) * 1000
        except Exception:
            time.sleep(2)
    return None, None


def decision(q, a, lab):
    """-> (correct, confidence = max prob, n_options)"""
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        p = [a["probabilities"][k] for k in keys]
        return a["choice"] == lab, max(p), len(keys)
    if t == "noul":
        return (a["noul"] > 0.5) == bool(lab), max(a["noul"], 1 - a["noul"]), 2
    p = [a["probabilities"][str(i)] for i in range(len(q["criteria"]))]
    return round(a["score"]) == int(lab), max(p), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--eval", default="/work/thai/data/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-options", type=int, default=10, help="questions with more options always go to the teacher")
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.eval, encoding="utf-8")]

    student = laya.Agent(args.student, device="cuda")
    rows = []  # per decision
    s_lat = {}
    for r in records:
        t = time.perf_counter()
        try:
            ans = student.predict(r["state"], r["questions"])["answers"]
        except Exception:
            ans = None
        s_lat[r["id"]] = (time.perf_counter() - t) * 1000
        for qid, q in r["questions"].items():
            if qid not in r["labels"]:
                continue
            if ans is None:
                rows.append({"id": r["id"], "qid": qid, "source": r["source"], "type": q["type"], "s_ok": None, "s_conf": 0.0, "k": 0})
            else:
                ok, conf, k = decision(q, ans[qid], r["labels"][qid])
                rows.append({"id": r["id"], "qid": qid, "source": r["source"], "type": q["type"], "s_ok": bool(ok), "s_conf": float(conf), "k": k})
    del student

    with ThreadPoolExecutor(8) as ex:
        t_res = dict(zip([r["id"] for r in records], ex.map(lambda r: teacher_call(args.teacher, r), records)))
    t_ok = {}
    t_lat = {}
    for r in records:
        ans, ms = t_res[r["id"]]
        t_lat[r["id"]] = ms
        for qid, q in r["questions"].items():
            if qid in r["labels"] and ans is not None:
                t_ok[(r["id"], qid)] = bool(decision(q, ans[qid], r["labels"][qid])[0])
    for row in rows:
        row["t_ok"] = t_ok.get((row["id"], row["qid"]))
        row["s_ms"] = s_lat[row["id"]]
        row["t_ms"] = t_lat[row["id"]]

    usable = [x for x in rows if x["t_ok"] is not None]
    n = len(usable)
    mean_s = sum(s_lat.values()) / len(s_lat)
    mean_t = sum(v for v in t_lat.values() if v) / max(1, sum(1 for v in t_lat.values() if v))
    curve = []
    for thr in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]:
        to_teacher = [x for x in usable if x["s_ok"] is None or x["s_conf"] < thr or x["k"] > args.max_options]
        keep = [x for x in usable if x not in to_teacher]
        acc = (sum(x["s_ok"] for x in keep) + sum(x["t_ok"] for x in to_teacher)) / n
        frac = len(to_teacher) / n
        curve.append({"threshold": thr, "accuracy": round(acc, 4), "teacher_fraction": round(frac, 4),
                      "student_acc_on_kept": round(sum(x["s_ok"] for x in keep) / max(1, len(keep)), 4),
                      "teacher_acc_on_sent": round(sum(x["t_ok"] for x in to_teacher) / max(1, len(to_teacher)), 4),
                      "est_mean_ms": round(mean_s + frac * mean_t, 1)})
    summary = {"decisions": n, "student_only": round(sum(x["s_ok"] for x in usable) / n, 4), "teacher_only": round(sum(x["t_ok"] for x in usable) / n, 4),
               "student_ms_per_record": round(mean_s, 1), "teacher_ms_per_record": round(mean_t, 1), "max_options": args.max_options, "curve": curve}
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))
    print(f"{'thr':>5s} {'acc':>6s} {'to_teacher':>10s} {'student@kept':>12s} {'teacher@sent':>12s} {'est ms':>7s}")
    for c in curve:
        print(f"{c['threshold']:5.2f} {c['accuracy']:6.3f} {c['teacher_fraction']:10.3f} {c['student_acc_on_kept']:12.3f} {c['teacher_acc_on_sent']:12.3f} {c['est_mean_ms']:7.0f}")
    json.dump({**summary, "rows": rows}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

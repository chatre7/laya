"""Evaluate a laya checkpoint (HF id or local dir) on the Thai eval records from prep_thai.py
plus the 5 hand-written ticket cases, and print per-source metrics.

    python eval_thai.py --model convaiinnovations/laya-multilingual --eval /work/thai/data/eval.jsonl --out base.json
    python eval_thai.py --model /work/thai/out/laya-th --eval /work/thai/data/eval.jsonl --out ft.json
"""
import argparse
import json
import time
from collections import defaultdict

import laya

TICKET_Q = {
    "department": {"type": "choice", "instructions": "ทีมใดควรรับผิดชอบ",
                   "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้ การชำระเงิน", "technical": "ระบบหรือสัญญาณใช้งานไม่ได้", "sales": "สมัคร เปลี่ยนแพ็กเกจ โปรโมชั่น"}},
    "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"},
}
TICKETS = [
    ("โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ โทรไปสามรอบแล้วไม่มีใครรับ", "billing", 2, True),
    ("เน็ตบ้านหลุดบ่อยมากตั้งแต่เมื่อคืน รีสตาร์ทเราเตอร์แล้วก็ยังไม่ได้ ช่วยดูให้หน่อยครับ", "technical", 1, False),
    ("สนใจเปลี่ยนเป็นแพ็กเกจ 5G ที่โปรโมชั่นลด 50% ต้องทำยังไงบ้างคะ", "sales", 0, False),
    ("ใบแจ้งหนี้เดือนนี้แพงกว่าปกติ ขอทราบรายละเอียดค่าใช้จ่ายหน่อยค่ะ", "billing", 0, False),
    ("แอปล็อกอินไม่ได้ ขึ้นว่ารหัสผิดทั้งที่ถูก โกรธมากแล้วนะ จะยกเลิกบริการ", "technical", 2, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="/work/thai/data/eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--teacher", default="", help="distill_eval.jsonl: also report student-vs-teacher agreement")
    args = ap.parse_args()
    agent = laya.Agent(args.model, device=args.device)

    teacher = None
    if args.teacher:
        agree = defaultdict(lambda: {"n": 0, "argmax": 0, "tv": 0.0})
        for line in open(args.teacher, encoding="utf-8"):
            r = json.loads(line)
            try:
                res = agent.predict(r["state"], r["questions"])
            except Exception:
                continue
            for qid, q in r["questions"].items():
                a, t = res["answers"][qid], r["targets"][qid]
                if q["type"] == "choice":
                    p = [a["probabilities"][k] for k in q["criteria"]]
                elif q["type"] == "score":
                    p = [a["probabilities"][str(i)] for i in range(len(q["criteria"]))]
                else:
                    p = [1 - a["noul"], a["noul"]]
                s = agree[q["type"]]
                s["n"] += 1
                s["argmax"] += max(range(len(p)), key=p.__getitem__) == max(range(len(t)), key=t.__getitem__)
                s["tv"] += sum(abs(x - y) for x, y in zip(p, t)) / 2  # total variation distance
        teacher = {k: {"n": v["n"], "argmax_agreement": round(v["argmax"] / max(1, v["n"]), 4), "mean_tv": round(v["tv"] / max(1, v["n"]), 4)}
                   for k, v in agree.items()}
        print(f"student vs teacher on {args.teacher}:")
        for k, v in sorted(teacher.items()):
            print(f"  {k:7s} n={v['n']:5d} argmax agreement {v['argmax_agreement']:.3f}  mean TV distance {v['mean_tv']:.3f}")

    stats = defaultdict(lambda: {"n": 0, "correct": 0, "abs_err": 0.0, "type": ""})
    lat = []
    for line in open(args.eval, encoding="utf-8"):
        r = json.loads(line)
        t = time.perf_counter()
        try:
            res = agent.predict(r["state"], r["questions"])
        except Exception as e:  # e.g. options exceed head budget
            stats[f'{r["source"]}:error']["n"] += 1
            continue
        lat.append((time.perf_counter() - t) * 1000)
        for qid, q in r["questions"].items():
            if qid not in r["labels"]:
                continue
            a, lab = res["answers"][qid], r["labels"][qid]
            key = f'{r["source"]}:{q["type"]}'
            s = stats[key]
            s["type"] = q["type"]
            s["n"] += 1
            if q["type"] == "choice":
                s["correct"] += a["choice"] == lab
            elif q["type"] == "noul":
                s["correct"] += (a["noul"] > 0.5) == bool(lab)
            else:
                s["correct"] += round(a["score"]) == int(lab)
                s["abs_err"] += abs(a["score"] - int(lab))

    tickets = {"department": 0, "refund": 0, "frustration_mae": 0.0, "rows": []}
    for state, dept, fr, refund in TICKETS:
        a = agent.predict(state, TICKET_Q)["answers"]
        tickets["department"] += a["department"]["choice"] == dept
        tickets["refund"] += (a["refund"]["noul"] > 0.5) == refund
        tickets["frustration_mae"] += abs(a["frustration"]["score"] - fr) / len(TICKETS)
        tickets["rows"].append({"state": state[:30], "dept": a["department"]["choice"], "p": a["department"]["probabilities"][a["department"]["choice"]],
                                "frustration": round(a["frustration"]["score"], 2), "refund": a["refund"]["noul"]})

    summary = {"model": args.model, "mean_ms": sum(lat) / max(1, len(lat)), "tickets": tickets, "teacher": teacher, "sources": {}}
    print(f"\n{args.model}: {len(lat)} eval records, {summary['mean_ms']:.0f} ms/record")
    print(f"{'source:type':28s} {'n':>5s} {'acc':>6s} {'mae':>6s}")
    for key in sorted(stats):
        s = stats[key]
        acc = s["correct"] / max(1, s["n"])
        mae = s["abs_err"] / max(1, s["n"]) if s["type"] == "score" else None
        summary["sources"][key] = {"n": s["n"], "acc": round(acc, 4), "mae": None if mae is None else round(mae, 4)}
        print(f"{key:28s} {s['n']:5d} {acc:6.3f} {'' if mae is None else f'{mae:6.3f}'}")
    print(f"tickets: department {tickets['department']}/5  refund {tickets['refund']}/5  frustration MAE {tickets['frustration_mae']:.2f}")
    for row in tickets["rows"]:
        print(f"   {row['state']:30s} {row['dept']:9s} p={row['p']:.2f} frustration={row['frustration']:.2f} refund={row['refund']:.2f}")
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

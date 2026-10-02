"""What merging the intents buys, with no retraining: on the hand-labelled real sets (Pantip long, short chat versions, Google
Play reviews) compare the fine-intent accuracy with the accuracy on the merged groups of intent_groups.py, where the group
answer is the argmax of the summed fine probabilities. Also prints the group pairs still confused and writes
INTENT_GROUPS.md, the sheet for the call-center team.

    python eval_groups.py --models /work/thai/out/laya-th-run14 --md /work/thai/cs/INTENT_GROUPS.md
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import INTENTS, intent_question  # noqa: E402
from intent_groups import GROUPS, group_of, group_probs  # noqa: E402

BIZ = ("telecom", "banking", "insurance")


def load(args):
    rows = [{"set": "long", "business": r["source"], "text": r["state"], "intent": r["labels"]["intent"]}
            for r in map(json.loads, open(args.long, encoding="utf-8"))]
    for s in map(json.loads, open(args.short, encoding="utf-8")):
        if s["set"] == "eval":
            rows.append({"set": "short", "business": s["business"], "text": s["text"], "intent": s["labels"]["intent"]})
    labels = json.load(open(args.play_labels, encoding="utf-8"))["labels"]
    for r in map(json.loads, open(args.play, encoding="utf-8")):
        if r["id"] in labels:
            rows.append({"set": "reviews", "business": r["business"], "text": r["text"], "intent": labels[r["id"]][0]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--long", default="/work/thai/data/domain/real_cs_eval.jsonl")
    ap.add_argument("--short", default="/work/thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--play", default="/work/thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--play-labels", default="/work/thai/data_domain/play_labels.json")
    ap.add_argument("--out", default="/work/thai/out/groups.json")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    rows = load(args)
    report = {}
    for m in args.models.split(","):
        agent = laya.Agent(m, device="cuda")
        st, conf = defaultdict(float), defaultdict(Counter)
        for r in rows:
            biz = r["business"]
            p = agent.predict(r["text"], {"intent": intent_question(biz)})["answers"]["intent"]["probabilities"]
            fine = max(p, key=p.get)
            gp = group_probs(biz, p)
            g_pred, g_gold = max(gp, key=gp.get), group_of(biz, r["intent"])
            top3 = sorted(gp, key=gp.get, reverse=True)[:3]
            for key in ((r["set"], biz), (r["set"], "all"), ("all", biz), ("all", "all")):
                st[key, "n"] += 1
                st[key, "fine"] += fine == r["intent"]
                st[key, "group"] += g_pred == g_gold
                st[key, "group3"] += g_gold in top3
            if g_pred != g_gold:
                conf[biz][f"{g_gold} -> {g_pred}"] += 1
        del agent
        name = os.path.basename(m)
        report[name] = {"rows": {}, "confusions": {b: conf[b].most_common(10) for b in BIZ}}
        print(f"\n== {name}   groups: " + ", ".join(f"{b} {len(INTENTS[b])} intents -> {len(GROUPS[b])}" for b in BIZ))
        print(f"{'set':8s} {'business':10s} {'n':>5s} {'fine':>6s} {'group':>6s} {'top-3':>6s}")
        for sset in ("long", "short", "reviews", "all"):
            for biz in BIZ + ("all",):
                n = st[(sset, biz), "n"]
                if not n:
                    continue
                line = {"n": int(n), "fine": round(st[(sset, biz), "fine"] / n, 4), "group": round(st[(sset, biz), "group"] / n, 4),
                        "group_top3": round(st[(sset, biz), "group3"] / n, 4)}
                report[name]["rows"][f"{sset}_{biz}"] = line
                print(f"{sset:8s} {biz:10s} {line['n']:5d} {line['fine']:6.3f} {line['group']:6.3f} {line['group_top3']:6.3f}")
        for biz in BIZ:
            print(f"{biz} still confused (gold -> answer): {conf[biz].most_common(8)}")
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)

    if args.md:
        name = os.path.basename(args.models.split(",")[-1])
        rep = report[name]
        th = {"telecom": "ค่ายมือถือ / อินเทอร์เน็ต", "banking": "ธนาคาร", "insurance": "ประกัน"}
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("# ร่างรายการ \"เรื่องที่ลูกค้าติดต่อ\" แบบยุบแล้ว (สำหรับทีม call center ตรวจ)\n\n"
                    "หลักการ: สองเรื่องถูกรวมเป็นหมวดเดียวเมื่อ **ส่งไปทีมเดียวกันและพนักงานทำขั้นตอนเดียวกัน** "
                    "ขอให้ช่วยดูว่า (1) หมวดไหนควรแยกเพราะทำงานต่างกันจริง (2) หมวดไหนควรรวมกันอีก (3) มีเรื่องที่เจอบ่อยแต่ไม่มีในรายการไหม\n\n"
                    f"ตัวเลขวัดจากข้อความจริงที่ติดป้ายด้วยมือ {int(st[('all', 'all'), 'n'])} ข้อ (โพสต์ Pantip ยาว, ฉบับย่อแบบแชท, รีวิวแอป) ด้วยโมเดล {name} "
                    "โดยไม่ได้เทรนใหม่ ไฟล์นี้สร้างจาก `intent_groups.py` ด้วย `eval_groups.py`\n\n")
            f.write("| ธุรกิจ | จำนวนเรื่องเดิม | หลังยุบ | ทายถูก (เดิม) | ทายถูก (หลังยุบ) | คำตอบถูกอยู่ใน 3 อันดับแรก |\n|---|---|---|---|---|---|\n")
            for biz in BIZ:
                r = rep["rows"][f"all_{biz}"]
                f.write(f"| {th[biz]} | {len(INTENTS[biz])} | {len(GROUPS[biz])} | {r['fine']:.0%} | {r['group']:.0%} | {r['group_top3']:.0%} |\n")
            for biz in BIZ:
                f.write(f"\n## {th[biz]}\n\n| หมวด | คำอธิบาย | เรื่องเดิมที่รวมอยู่ในหมวดนี้ |\n|---|---|---|\n")
                for g, (desc, members) in GROUPS[biz].items():
                    f.write(f"| `{g}` | {desc} | " + "<br>".join(f"{INTENTS[biz][i]} (`{i}`)" for i in members) + " |\n")
                f.write("| `other` | ไม่เข้าหมวดใด หรือไม่ใช่เรื่องติดต่อฝ่ายบริการลูกค้า | |\n")
                f.write("\nหมวดที่โมเดลยังสับสนบ่อย (เฉลย -> ที่โมเดลตอบ, จำนวนครั้ง): " + ", ".join(f"`{k}` {v}" for k, v in rep["confusions"][biz][:6]) + "\n")
        print("wrote", args.md)


if __name__ == "__main__":
    main()

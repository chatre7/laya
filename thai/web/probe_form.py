"""Replay the narrow form-agent questions (System One demo/form_agent.py --dump, answered by the teacher on 8/8 passing tasks)
against laya checkpoints. Only questions whose teacher answer is right by construction are kept: which input takes a value,
which select/radio option matches it, which pop-up suggestion matches it, which button submits.

    python probe_form.py --dump /work/thai/data/web/form_teacher_decisions.jsonl --device cpu
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402

KINDS = {"ควรกรอกหรือเลือก": "field", "ตัวเลือกไหนตรงกับ": "option", "รายการที่เด้งขึ้นมา": "suggestion", "ควรกดปุ่มไหนเพื่อ": "submit"}
MODELS = ["convaiinnovations/laya-multilingual", "/work/thai/out/laya-th-run3", "/work/thai/out/laya-th-run8",
          "/work/thai/out/laya-th-run10", "/work/thai/out/laya-th-web1"]


def kind(r):
    ins = r["questions"]["q"]["instructions"]
    return next((v for k, v in KINDS.items() if ins.startswith(k)), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rows = [r for r in map(json.loads, open(args.dump, encoding="utf-8")) if kind(r)]
    print(f"{len(rows)} questions: {dict(Counter(kind(r) for r in rows))}")
    for m in MODELS:
        try:
            agent = laya.Agent(m, device=args.device)
        except Exception as e:  # noqa: BLE001
            print(f"{m}: cannot load ({type(e).__name__})")
            continue
        ok, n, wrong = Counter(), Counter(), []
        for r in rows:
            c = agent.predict(r["state"], r["questions"])["answers"]["q"]["choice"]
            k = kind(r)
            n[k] += 1
            ok[k] += c == r["answer"]
            if c != r["answer"] and len(wrong) < 4:
                wrong.append(f'{r["questions"]["q"]["instructions"][:50]} -> {c[:45]} (teacher: {r["answer"][:45]})')
        print(f"\n{m}: {sum(ok.values())}/{len(rows)}  " + "  ".join(f"{k} {ok[k]}/{n[k]}" for k in n))
        for w in wrong:
            print("   ", w)
        del agent


if __name__ == "__main__":
    main()

"""The LLM labeller (label_play_llm.py, Qwen3-8B) on forum / social posts: --eval scores it on the 360 hand-labelled Pantip
posts (real_cs_eval.jsonl); otherwise it labels every Pantip post that has no hand label yet, and the wisesight posts that
mention the three businesses.

    python3 thai/cs/label_pantip_llm.py --eval
    python3 thai/cs/label_pantip_llm.py --out thai/data/cs/pantip_llm.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_play_llm import ask  # noqa: E402

BIZ = ("telecom", "banking", "insurance")


def key_of(t):
    return " ".join(t.split())[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--domain", default="thai/data/domain")
    ap.add_argument("--hand", default="thai/data_domain", help="hand-labelled batches (pantip_<biz>_extra*.jsonl, review samples)")
    ap.add_argument("--out", default="thai/data/cs/pantip_llm.jsonl")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()
    ev = [json.loads(l) for l in open(os.path.join(args.domain, "real_cs_eval.jsonl"), encoding="utf-8")]
    if args.eval:
        rows = [{"id": r["id"], "business": r["source"], "text": r["state"], "gold": r["labels"]} for r in ev]
    else:
        seen = {key_of(r["state"]) for r in ev}
        for p in glob.glob(os.path.join(args.hand, "pantip_*_extra*.jsonl")) + glob.glob(os.path.join(args.hand, "pantip_*_review_sample.jsonl")):
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    seen.add(key_of(json.loads(l)["text"]))
        rows = []
        for src in ("pantip", "wisesight"):
            for biz in BIZ:
                p = os.path.join(args.domain, f"{src}_{biz}.jsonl")
                if not os.path.exists(p):
                    continue
                for l in open(p, encoding="utf-8"):
                    t = json.loads(l).get("text", "").strip()
                    k = key_of(t)
                    if len(t) >= 10 and k not in seen:
                        seen.add(k)
                        rows.append({"id": f"{src}llm-{biz}-{len(rows)}", "business": biz, "source": src, "text": t})
        print(f"{len(rows)} posts without a hand label: {dict(Counter((r['source'], r['business']) for r in rows))}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda r: ask(args.url, args.model, r["business"], " ".join(r["text"].split()), args.max_chars), rows))
    print(f"{sum(o is not None for o in outs)}/{len(rows)} labelled in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.eval:
        st, wrong = Counter(), Counter()
        for r, o in zip(rows, outs):
            g = r["gold"]
            o = o or ["other", "support", 0]
            for key in (r["business"], "all"):
                st[key, "n"] += 1
                st[key, "intent"] += o[0] == g["intent"]
                st[key, "department"] += o[1] == g["department"]
                st[key, "urgency"] += o[2] == g["urgency"]
                if g["intent"] != "other":
                    st[key, "n_in"] += 1
                    st[key, "false_other"] += o[0] == "other"
                else:
                    st[key, "n_other"] += 1
                    st[key, "other_caught"] += o[0] == "other"
            if o[0] != g["intent"]:
                wrong[f"{g['intent']} -> {o[0]}"] += 1
        for key in BIZ + ("all",):
            n = st[key, "n"]
            print(f"{key:10s} n={n:4d} intent {st[key, 'intent'] / n:.3f} dept {st[key, 'department'] / n:.3f} urg {st[key, 'urgency'] / n:.3f} "
                  f"false-other {st[key, 'false_other']}/{st[key, 'n_in']} other-caught {st[key, 'other_caught']}/{st[key, 'n_other']}")
        print("intent errors (gold -> llm):", wrong.most_common(12))
        with open(args.out.replace(".jsonl", "_eval.jsonl"), "w", encoding="utf-8") as f:  # for agree_check.py
            for r, o in zip(rows, outs):
                f.write(json.dumps({"id": r["id"], "business": r["business"], "text": r["text"], "gold": r["gold"], "llm": o}, ensure_ascii=False) + "\n")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            if o is not None:
                f.write(json.dumps({**r, "labels": {"intent": o[0], "department": o[1], "urgency": o[2]}}, ensure_ascii=False) + "\n")
    print("intents", Counter(o[0] for o in outs if o).most_common(15))


if __name__ == "__main__":
    main()

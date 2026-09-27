"""krathu-500 (Pittawat2542/krathu-500, Pantip comments with POS/NEG/NEU labels, no licence: internal eval only) ->
eval_thai.py-format sentiment records: all POS and NEG comments + the same number of NEU, one 3-way choice question.
Optionally scores HTTP endpoints on it (same as build_real_eval.py).

    python build_krathu_eval.py --raw thai/data_domain/krathu500_raw.csv --out thai/data_domain/krathu500_eval.jsonl --model teacher=http://172.18.72.145:8010
"""
import argparse
import csv
import json
import random
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

Q = {"type": "choice", "instructions": "อารมณ์โดยรวมของข้อความ",
     "criteria": {"positive": "ชม ขอบคุณ ให้กำลังใจ มองโลกในแง่ดี", "negative": "ด่า โกรธ เศร้า บ่น มองโลกในแง่ร้าย",
                  "neutral": "เล่าเฉย ๆ ให้ข้อมูล ถาม หรือปนกันจนบอกไม่ได้"}}
LAB = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}


def ask(url, r):
    body = json.dumps({"state": r["state"], "questions": r["questions"], "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=url")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in csv.DictReader(open(args.raw, encoding="utf-8-sig")) if len(r["text"].strip()) >= 5 and r["class_label"] in LAB]
    by = {k: [r for r in rows if r["class_label"] == k] for k in LAB}
    n = min(len(by["POS"]), len(by["NEG"]))
    keep = by["POS"] + by["NEG"] + rng.sample(by["NEU"], n)
    rng.shuffle(keep)
    recs = [{"id": f"krathu-{r['comment_id']}", "source": "krathu500", "state": " ".join(r["text"].split()), "questions": {"sentiment": Q},
             "labels": {"sentiment": LAB[r["class_label"]]}} for r in keep]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} records {dict(Counter(r['labels']['sentiment'] for r in recs))} -> {args.out}")
    for spec in args.model:
        name, url = spec.split("=", 1)
        with ThreadPoolExecutor(8) as ex:
            answers = list(ex.map(lambda r: ask(url, r), recs))
        ok, tot, conf = 0, 0, Counter()
        for r, a in zip(recs, answers):
            if a is None:
                continue
            tot += 1
            pred = a["sentiment"]["choice"]
            ok += pred == r["labels"]["sentiment"]
            conf[(r["labels"]["sentiment"], pred)] += 1
        print(f"== {name}: accuracy {ok / tot:.3f} on {tot}; (label, prediction) counts {dict(conf)}")


if __name__ == "__main__":
    main()

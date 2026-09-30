"""Google Play reviews (Thai) of Thai banking / telecom / insurance apps: short, real, in the customer's own words. Apps are
found by search so package ids need not be known; reviews are fetched newest first, Thai only, deduplicated, and a held-out
sample is split off for hand labelling.

    python fetch_play_reviews.py --out thai/data_domain/play_reviews.jsonl --per-app 2500 --eval 300
"""
import argparse
import json
import random
import re
import time

from google_play_scraper import Sort, reviews, search
from google_play_scraper import app as gp_app

# package ids where known (search picked K PLUS Vietnam, merchant apps and a leasing company for the plain names); otherwise a
# search query. Wallets and consumer-loan apps count as banking: the questions are the same (transfers, top-ups, fees, loans).
APPS = {
    "banking": ["com.kasikorn.retail.mbanking.wap", "com.scb.phone", "ktbcs.netbank", "com.bbl.mobilebanking", "com.krungsri.kma",
                "com.ttbbank.oneapp.android", "com.gsb.mymo", "com.uob.mighty.app", "th.co.truemoney.wallet", "com.ktb.customer.qr",
                "co.th.muangthaileasing.mtls", "KTC Mobile บัตรเครดิต", "CardX"],
    "telecom": ["com.ais.mimo.eservice", "com.truelife.mobile.android.trueiservice", "th.co.mimotech.android.neweasyappais", "com.ntsuperapp",
                "com.ttbb.ota", "dtac app ดีแทค", "3BB Member", "AIS Fibre", "TOT"],
    "insurance": ["global.fwd.omne", "com.axa.app.myaxa.th", "com.nextzy.tipapp", "com.easysunday.app", "AIA+ ประกันชีวิต", "MTL Click",
                  "Thai Life Insurance ไทยประกันชีวิต", "Viriyah Insurance วิริยะ", "Krungthai-AXA", "Allianz Ayudhya", "Roojai ประกันรถ",
                  "Bangkok Insurance", "Muang Thai Insurance", "Dhipaya"],
}


def thai_ok(t):
    thai = sum(1 for ch in t if "฀" <= ch <= "๿")
    alpha = sum(1 for ch in t if ch.isalpha())
    return len(t) >= 10 and thai >= 0.6 * max(1, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="thai/data_domain/play_reviews.jsonl")
    ap.add_argument("--eval-out", default="thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--per-app", type=int, default=2500)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, seen = [], set()
    for biz, names in APPS.items():
        for name in names:
            try:
                if "." in name and " " not in name:  # a package id
                    a = gp_app(name, lang="th", country="th")
                    app = {"appId": name, "title": a["title"]}
                else:  # the first hit is sometimes a collection entry without an appId: take the first real app
                    hits = [h for h in search(name, lang="th", country="th", n_hits=6) if h.get("appId")]
                    if not hits:
                        print(f"no app for {name}")
                        continue
                    app = hits[0]
            except Exception as e:  # noqa: BLE001
                print(f"lookup failed {name}: {type(e).__name__}")
                continue
            if app["appId"] in {r["app_id"] for r in rows}:
                print(f"{biz:9s} {name}: same app as before ({app['appId']}), skipped")
                continue
            got, token, n = 0, None, 0
            while got < args.per_app:
                try:
                    batch, token = reviews(app["appId"], lang="th", country="th", sort=Sort.NEWEST, count=min(200, args.per_app - got),
                                           continuation_token=token)
                except Exception as e:  # noqa: BLE001
                    print(f"reviews failed {app['appId']}: {type(e).__name__}")
                    break
                for r in batch:
                    got += 1
                    t = " ".join((r.get("content") or "").split())
                    k = re.sub(r"\W+", "", t)[:80]
                    if not thai_ok(t) or k in seen:
                        continue
                    seen.add(k)
                    rows.append({"id": f"play-{biz}-{len(rows)}", "business": biz, "app": app["title"], "app_id": app["appId"], "score": r.get("score"),
                                 "at": str(r.get("at"))[:10], "text": t})
                    n += 1
                if not batch or token is None:
                    break
                time.sleep(0.3)
            print(f"{biz:9s} {app['title'][:40]:40s} {str(app['appId']):40s} fetched {got:5d} kept {n:5d}", flush=True)
    rng.shuffle(rows)
    ev = rows[: args.eval]
    tr = rows[args.eval:]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in rows:
        by[r["business"]] = by.get(r["business"], 0) + 1
    print(f"{len(rows)} reviews {by}; {len(ev)} held out for hand labels -> {args.eval_out}; {len(tr)} -> {args.out}")
    print("mean length", sum(len(r["text"]) for r in rows) / max(1, len(rows)))


if __name__ == "__main__":
    main()

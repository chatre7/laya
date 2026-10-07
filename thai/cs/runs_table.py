"""One table over the runs from the result files in thai/results: real posts (long / short), the 300 hand-labelled reviews,
mood against stars, the ambiguous set, cs9 held-out and the public set. A cell is "-" when that run was not measured on
that set (the sets were added over time). Prints Markdown; `--write` replaces the "## Run table" section of the README.

    python cs/runs_table.py --write
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
README = os.path.join(HERE, "..", "README.md")
WHAT = {
    3: "bigger distillation set + human labels, option budget 768",
    4: "call-center distillation",
    5: "calibration, more score items, an `other` intent",
    6: "telecom / banking / insurance / e-commerce",
    7: "in-register out-of-scope texts",
    8: "first real Thai in-domain text (Pantip), served 09-28 to 10-02",
    9: "1,000 more hand-labelled Pantip rows",
    10: "short second pass on the real rows",
    11: "short chat versions, human frustration labels, hard examples",
    12: "Google Play reviews, teacher + student agreement labels",
    13: "reviews labelled by Qwen3-8B",
    14: "Qwen labels on forum posts where the student agrees, served 10-02 to 10-03",
    15: "composed sarcasm + star-labelled mood; neutral collapsed, not served",
    16: "15 + Wisesight neutral / question, served 10-03 to 10-07",
    17: "16 with neutral x2; long-post triage down, not served",
    18: "16 + hagsmand1/laya-thai-decisions Thai split, not served",
    19: "16 + churn threat and contact effort from Qwen3-8B rules, served since 10-07",
}


def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def pct(x, d=0):
    return "-" if x is None else f"{100 * x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    runs = {}
    for p in glob.glob(os.path.join(RES, "real*.json")) + glob.glob(os.path.join(RES, "play*.json")) + glob.glob(os.path.join(RES, "stars*.json")):
        d = load(p)
        if not d:
            continue
        for key, v in d.items():
            m = re.match(r"laya-th-run(\d+)$", key)
            if not m or not isinstance(v, dict):
                continue
            r = runs.setdefault(int(m.group(1)), {})
            if os.path.basename(p).startswith("real"):
                r["long"], r["short"] = v.get("long_all"), v.get("short_all")
            elif os.path.basename(p).startswith("play"):
                r["play"] = v.get("all")
            else:
                r["stars"] = v.get("sentiment_matches_stars")
                r["amb"] = (v.get("ambiguous") or {}).get("accuracy")
    for p in glob.glob(os.path.join(RES, "run*_cs9.json")):
        n = int(os.path.basename(p)[3:].split("_")[0])
        d = load(p)
        if d:
            runs.setdefault(n, {})["cs9"] = d["overall"]["accuracy"]
    for p in glob.glob(os.path.join(RES, "run*.json")):
        m = re.match(r"run(\d+)\.json$", os.path.basename(p))
        if m:
            d = load(p)
            if d and "overall" in d:
                runs.setdefault(int(m.group(1)), {})["public"] = d["overall"]["accuracy"]
    head = ("| run | what changed | long posts: intent / dept / urgency | short: intent / dept / urgency | reviews: intent / dept / urgency | "
            "stars | ambiguous | cs9 | public |\n|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for n in sorted(runs):
        r = runs[n]
        t = lambda s: f"{pct(s['intent'])} / {pct(s['department'])} / {pct(s['urgency'])}" if s else "-"  # noqa: E731
        rows.append(f"| {n} | {WHAT.get(n, '')} | {t(r.get('long'))} | {t(r.get('short'))} | {t(r.get('play'))} | {pct(r.get('stars'))} | "
                    f"{pct(r.get('amb'))} | {pct(r.get('cs9'), 1)} | {pct(r.get('public'), 1)} |")
    table = head + "\n".join(rows) + "\n"
    print(table)
    if args.write:
        s = open(README, encoding="utf-8").read()
        sec = ("## Run table\n\nEvery run on every set it was measured on, in percent, from `results/` (`cs/runs_table.py --write`). "
               "Long = 360 hand-labelled Pantip posts, short = their 714 chat versions, reviews = 300 hand-labelled Google Play reviews, "
               "stars = 1-2 stars called negative or 4-5 positive (3,000 reviews, from run 15), ambiguous = 3,300 sentences of "
               "Tippawan/thai-ambiguous-sentiment, cs9 = 11,720 held-out decisions of the synthetic call-center set, public = the 2,455 "
               "public-dataset decisions. Differences under about 3 points on the 360-post set are noise.\n\n" + table + "\n")
        if "## Run table" in s:
            s = re.sub(r"## Run table\n.*?(?=\n## )", sec.rstrip("\n") + "\n", s, count=1, flags=re.S)
        else:
            s = s.replace("## Known limits of laya for our use", sec + "## Known limits of laya for our use", 1)
        open(README, "w", encoding="utf-8", newline="\n").write(s)
        print("written")


if __name__ == "__main__":
    main()

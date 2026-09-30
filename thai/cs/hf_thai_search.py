"""Thai datasets on the Hub that could feed the call-center model. Two lists: keyword hits whose id or tags say Thai (the
multilingual corpora that merely include Thai are skipped), and everything from the Thai NLP organisations. Prints downloads,
licence, size, task; --peek shows the first rows of the named datasets.

    python hf_thai_search.py
    python hf_thai_search.py --peek pythainlp/wisesight_sentiment,airesearch/...
"""
import argparse
import json
import re
from huggingface_hub import HfApi, HfFileSystem

QUERIES = ["thai customer", "thai call center", "thai complaint", "thai review", "thai chat", "thai intent", "thai support", "thai bank",
           "thai telecom", "thai insurance", "thai dialogue", "thai conversation", "thai sentiment", "thai instruct", "thai classification",
           "thai faq", "thai question", "thai ticket", "thai email", "thai sms", "thai social", "thai text", "thai nlp", "thai corpus",
           "pantip", "wisesight", "wongnai", "kbtg", "typhoon", "openthaigpt", "ภาษาไทย", "ลูกค้า", "ร้องเรียน", "รีวิว", "ธนาคาร", "ประกัน"]
ORGS = ["pythainlp", "airesearch", "SEACrowd", "iapp", "typhoon-ai", "scb10x", "nectec", "Thaweewat", "ThaiSC", "kbtg-labs", "openthaigpt",
        "Patt", "chulalongkorn", "wangchanberta", "vistec", "Porameht", "pakphum", "ZombitX64", "Chanon", "SuperAI"]
BIG = re.compile(r"c4|wikipedia|common_voice|oscar|culturax|fineweb|hplt|xnli|flores|belebele|opus|glotcc|megawika|ccnews|xp3|miracl|mldr|subscene",
                 re.I)


def thai_named(d):
    tags = d.tags or []
    return ("language:th" in tags and not BIG.search(d.id)) or re.search(r"thai|_th\b|-th\b|/th[-_]|pantip|wisesight|wongnai", d.id, re.I)


def row(d):
    tags = d.tags or []
    lic = next((t[8:] for t in tags if t.startswith("license:")), "?")
    size = next((t[16:] for t in tags if t.startswith("size_categories:")), "?")
    task = ",".join(t[16:] for t in tags if t.startswith("task_categories:"))[:36]
    return (d.downloads or 0, d.id, lic, size, task)


def show(title, rows, n=80):
    rows = sorted(set(rows), reverse=True)
    print(f"\n=== {title}: {len(rows)}")
    print(f"{'dl':>7} {'id':58s} {'licence':18s} {'size':10s} task")
    for dl, i, lic, size, task in rows[:n]:
        print(f"{dl:>7} {i:58s} {lic[:18]:18s} {size:10s} {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", default="")
    args = ap.parse_args()
    api = HfApi()
    if args.peek:
        fs = HfFileSystem()
        for i in args.peek.split(","):
            try:
                files = [f for f in api.list_repo_files(i, repo_type="dataset") if f.endswith((".jsonl", ".json", ".csv", ".parquet", ".tsv", ".txt"))]
                f = files[0]
                p = f"datasets/{i}/{f}"
                if f.endswith(".parquet"):
                    import pyarrow.parquet as pq
                    t = pq.ParquetFile(p, filesystem=fs)
                    print(f"\n## {i} {f} rows={t.metadata.num_rows} cols={t.schema_arrow.names}")
                    print("  ", json.dumps(t.read_row_group(0).slice(0, 3).to_pylist(), ensure_ascii=False)[:700])
                else:
                    with fs.open(p, "r", encoding="utf-8") as fh:
                        print(f"\n## {i} {f} ({len(files)} files)\n   {fh.read(700)}")
            except Exception as e:  # noqa: BLE001
                print(f"\n## {i}: cannot peek ({type(e).__name__}: {str(e)[:80]})")
        return
    hits = []
    for q in QUERIES:
        try:
            hits += [row(d) for d in api.list_datasets(search=q, limit=50) if thai_named(d)]
        except Exception as e:  # noqa: BLE001
            print("search failed", q, type(e).__name__)
    show("keyword hits, Thai-named", hits)
    org = []
    for o in ORGS:
        try:
            org += [row(d) for d in api.list_datasets(author=o, limit=100)]
        except Exception as e:  # noqa: BLE001
            print("org failed", o, type(e).__name__)
    show("Thai organisations", org, 120)


if __name__ == "__main__":
    main()

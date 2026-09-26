"""banking77 intent names/counts (mteb mirror) and the bank-complaint label set, via the datasets-server rows API."""
import collections
import json
import urllib.request


def rows(ds, config, split, offset, length=100):
    try:
        return json.load(urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={config}&split={split}&offset={offset}&length={length}", timeout=60))
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:100]}


def splits(ds):
    try:
        return json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/splits?dataset={ds}", timeout=60)).get("splits", [])
    except Exception as e:  # noqa: BLE001
        return []


for ds in ("mteb/banking77", "legacy-datasets/banking77"):
    sp = splits(ds)
    print("==", ds, [(s["config"], s["split"]) for s in sp][:6])
    if not sp:
        continue
    c, s = sp[0]["config"], sp[0]["split"]
    r = rows(ds, c, s, 0, 5)
    print("   features:", [f["name"] for f in r.get("features", [])])
    for row in r.get("rows", [])[:3]:
        print("     ", {k: str(v)[:100] for k, v in row["row"].items()})
    names = None
    for f in r.get("features", []):
        t = f.get("type", {})
        if isinstance(t, dict) and "names" in t:
            names = t["names"]
    cnt = collections.Counter()
    off, n = 0, 0
    while off < 10003:
        rr = rows(ds, c, s, off, 100)
        if "rows" not in rr:
            break
        for row in rr["rows"]:
            cnt[row["row"].get("label", row["row"].get("label_text"))] += 1
            n += 1
        off += 100
    if names:
        print(f"   {len(names)} intents over {n} rows: " + ", ".join(f"{names[i] if isinstance(i, int) else i}({v})" for i, v in sorted(cnt.items(), key=lambda x: str(x[0]))))
    else:
        print(f"   {len(cnt)} labels over {n} rows: " + ", ".join(f"{k}({v})" for k, v in cnt.most_common(80)))
    break

ds = "KunalEsM/bank_complaint_intent_classifier"
cnt = collections.Counter()
off = 0
while off < 20000:
    rr = rows(ds, "default", "train", off, 100)
    if "rows" not in rr:
        break
    for row in rr["rows"]:
        cnt[row["row"]["label"]] += 1
    off += 100
print(f"\n== {ds}: {sum(cnt.values())} rows, {len(cnt)} labels")
for k, v in cnt.most_common(60):
    print(f"   {v:5d} {k}")

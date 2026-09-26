"""Search Hugging Face for Thai app reviews / forum posts / tweets (real customer voice) and print licence, size, sample rows."""
import json
import urllib.request

from huggingface_hub import HfApi

api = HfApi()
QUERIES = ["thai app review", "google play thai", "app store thai", "pantip", "thai forum", "thai tweet", "twitter thai", "thai social media",
           "thai bank app", "kbank", "scb", "true", "ais thai", "thai complaint", "thai feedback", "thai chat log", "thai conversation dataset",
           "thai sentiment", "thai review", "thai comment", "thai messenger", "thai line chat"]
seen = {}
for q in QUERIES:
    for d in api.list_datasets(search=q, limit=40):
        seen.setdefault(d.id, d)
rows = []
for d in seen.values():
    tags = " ".join(d.tags or []).lower()
    idl = d.id.lower()
    if "language:th" in tags or "thai" in idl or "pantip" in idl or "_th" in idl or "-th" in idl or "wisesight" in idl:
        rows.append((d.downloads or 0, d.id, [t for t in (d.tags or []) if t.startswith(("license:", "size_categories:"))]))
rows.sort(reverse=True)
print(f"{len(rows)} Thai-ish datasets")
for dl, i, t in rows[:60]:
    print(f"  {dl:>7} {i:62s} {' '.join(t)}")


def first_rows(ds):
    try:
        sp = json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/splits?dataset={ds}", timeout=30)).get("splits", [])
        if not sp:
            return None
        c, s = sp[0]["config"], sp[0]["split"]
        return json.load(urllib.request.urlopen(f"https://datasets-server.huggingface.co/first-rows?dataset={ds}&config={c}&split={s}", timeout=30))
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:80]}


print("\n== samples of the most promising")
for name in [i for _, i, _ in rows if any(k in i.lower() for k in ("review", "pantip", "tweet", "twitter", "comment", "feedback", "complaint", "chat", "app"))][:12]:
    r = first_rows(name)
    if not r or "rows" not in r:
        print(f"-- {name}: {r}")
        continue
    print(f"-- {name}: features {[f['name'] for f in r['features']]}")
    for row in r["rows"][:3]:
        print("     ", {k: str(v)[:90] for k, v in row["row"].items()})

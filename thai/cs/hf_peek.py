"""Peek at candidate datasets via the datasets-server API: features, sample rows, and (for banking77) the full intent list."""
import collections
import json
import sys
import urllib.request


def get(url):
    try:
        return json.load(urllib.request.urlopen(url, timeout=60))
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)[:100]}


def peek(ds, n=4, key_hint=None):
    sp = get(f"https://datasets-server.huggingface.co/splits?dataset={ds}").get("splits", [])
    if not sp:
        print(f"-- {ds}: no splits")
        return
    c, s = sp[0]["config"], sp[0]["split"]
    r = get(f"https://datasets-server.huggingface.co/first-rows?dataset={ds}&config={c}&split={s}")
    if "rows" not in r:
        print(f"-- {ds}: {r}")
        return
    print(f"-- {ds} [{c}/{s}] splits={[(x['config'], x['split']) for x in sp][:4]} features={[f['name'] for f in r['features']]}")
    for row in r["rows"][:n]:
        print("     ", {k: str(v)[:100] for k, v in row["row"].items()})


for ds in sys.argv[1:] or ["PolyAI/banking77", "deccan-ai/insuranceQA-v2", "rvpierre/insurance-qa-en", "gorkemsevinc/Customer_Support_on_Twitter",
                          "KunalEsM/bank_complaint_intent_classifier", "nraptisss/telecom-intent-config-sft-10k", "DeepPavlov/hwu64", "clinc/clinc_oos"]:
    peek(ds)

# banking77: full label names + counts through the rows endpoint (paged)
labels = collections.Counter()
names = None
off = 0
while off < 10003:
    r = get(f"https://datasets-server.huggingface.co/rows?dataset=PolyAI/banking77&config=default&split=train&offset={off}&length=100")
    if "rows" not in r:
        break
    if names is None:
        for f in r["features"]:
            if f["name"] == "label" and "names" in f.get("type", {}):
                names = f["type"]["names"]
    for row in r["rows"]:
        labels[row["row"]["label"]] += 1
    off += 100
if names:
    print(f"\nbanking77: {len(names)} intents, {sum(labels.values())} rows sampled")
    print(", ".join(f"{names[i]}({labels[i]})" for i in range(len(names))))

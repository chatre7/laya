"""Multimodal-Mind2Web (text columns only, range reads over hf://, no screenshots) -> element-choice records in the format of
System One's demo/browser_agent.py: each option is `[i] <Thai role> (<role>) "<name>" value=".."`, the positive element plus a
random number (3-19) of negative candidates that look interactive, shuffled.

    python prep_m2w.py --out /work/thai/data/web            # all splits
    python prep_m2w.py --out /work/thai/data/web --limit 50  # smoke
"""
import argparse
import json
import random
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "osunlp/Multimodal-Mind2Web"
COLS = ["action_uid", "cleaned_html", "operation", "pos_candidates", "neg_candidates", "website", "domain", "subdomain",
        "annotation_id", "confirmed_task", "action_reprs", "target_action_index"]
ROLE_TH = {"textbox": "ช่องกรอกข้อความ", "combobox": "เมนูเลือก", "link": "ลิงก์", "button": "ปุ่ม", "checkbox": "กล่องติ๊ก",
           "radio": "ปุ่มเลือก", "menuitem": "รายการเมนู", "tab": "แท็บ"}
INTERACTIVE = {"a", "button", "input", "select", "textarea", "option", "label"}


class NodeText(HTMLParser):
    """backend_node_id -> (tag, attrs, text of its <text> descendants)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.nodes = [], {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        bid = a.get("backend_node_id")
        self.stack.append(bid)
        if bid:
            self.nodes[bid] = {"tag": tag, "attrs": a, "text": []}

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        d = " ".join(data.split())
        if not d:
            return
        for bid in self.stack[-8:]:  # nearest ancestors only: a container's name is not the whole page
            if bid and bid in self.nodes and sum(len(t) for t in self.nodes[bid]["text"]) < 120:
                self.nodes[bid]["text"].append(d)


def role_of(tag, attrs):
    r = (attrs.get("role") or "").lower()
    if r in ROLE_TH:
        return r
    if tag == "a":
        return "link"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        t = (attrs.get("type") or "text").lower()
        return {"checkbox": "checkbox", "radio": "radio", "submit": "button", "button": "button", "image": "button"}.get(t, "textbox")
    return "button"  # clickable span/div/p/li/img: the demo snapshot calls these buttons too


def describe(cand, nodes):
    c = json.loads(cand)
    attrs = json.loads(c.get("attributes") or "{}")
    bid = c.get("backend_node_id") or attrs.get("backend_node_id")
    node = nodes.get(bid, {"tag": c.get("tag", ""), "attrs": {}, "text": []})
    a = {**node["attrs"], **attrs}
    tag = c.get("tag") or node["tag"]
    role = role_of(tag, a)
    name = " ".join(node["text"]) or a.get("aria_label") or a.get("aria-label") or a.get("placeholder") or a.get("title") \
        or a.get("alt") or a.get("value") or a.get("name") or ""
    name = re.sub(r"<[^>]*>", " ", str(name))  # some attributes carry escaped markup ("<text backend_node_id=..>Save</text>")
    name = re.sub(r"\s+", " ", name).strip().replace('"', "'")[:60]
    extra = f' value="{str(a.get("value", ""))[:30]}"' if role in ("textbox", "combobox") else ""
    looks = tag in INTERACTIVE or a.get("is_clickable") == "true" or a.get("role")
    return {"role": role, "name": name, "extra": extra, "tag": tag, "looks_interactive": bool(looks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data/web")
    ap.add_argument("--limit", type=int, default=0, help="rows per split (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True) if f.path.endswith(".parquet"))
    by_split = {}
    for f in files:
        by_split.setdefault(f.split("/")[-1].split("-")[0], []).append(f)
    for split, paths in sorted(by_split.items()):
        n, skipped, stats = 0, Counter(), Counter()
        with open(out / f"m2w_{split}.jsonl", "w", encoding="utf-8") as fo:
            for p in paths:
                pf = pq.ParquetFile(f"datasets/{REPO}/{p}", filesystem=fs)
                for rg in range(pf.num_row_groups):
                    for row in pf.read_row_group(rg, columns=COLS).to_pylist():
                        if args.limit and n >= args.limit:
                            break
                        if not row["pos_candidates"]:
                            skipped["no_positive"] += 1
                            continue
                        parser = NodeText()
                        try:
                            parser.feed(row["cleaned_html"] or "")
                        except Exception:  # noqa: BLE001
                            skipped["html"] += 1
                            continue
                        pos = describe(row["pos_candidates"][0], parser.nodes)
                        pos_s = f'{ROLE_TH[pos["role"]]} ({pos["role"]}) "{pos["name"]}"{pos["extra"]}'
                        negs, seen = [], {pos_s}
                        for c in row["neg_candidates"]:
                            d = describe(c, parser.nodes)
                            s = f'{ROLE_TH[d["role"]]} ({d["role"]}) "{d["name"]}"{d["extra"]}'
                            if not d["looks_interactive"] or not d["name"] or s in seen:
                                continue
                            seen.add(s)
                            negs.append(s)
                        k = rng.randint(3, 19)
                        opts = [pos_s] + rng.sample(negs, min(k, len(negs)))
                        rng.shuffle(opts)
                        opts = [f"[{i}] {s}" for i, s in enumerate(opts)]
                        target = next(o for o in opts if o.split("] ", 1)[1] == pos_s)
                        op = json.loads(row["operation"])
                        idx = int(row["target_action_index"])
                        rec = {"id": row["action_uid"], "split": split, "website": row["website"], "domain": row["domain"],
                               "subdomain": row["subdomain"], "annotation_id": row["annotation_id"], "task_en": row["confirmed_task"],
                               "previous": [re.sub(r"\s+", " ", a) for a in row["action_reprs"][:idx][-3:]],
                               "options": opts, "target": target, "op": op.get("op", "CLICK"), "value": op.get("value", "")}
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                        stats["options"] += len(opts)
                        stats["pos_named"] += bool(pos["name"])
                        stats[f"op_{rec['op']}"] += 1
                    if args.limit and n >= args.limit:
                        break
                if args.limit and n >= args.limit:
                    break
        print(f"{split}: {n} records, skipped {dict(skipped)}, mean options {stats['options'] / max(1, n):.1f}, "
              f"positive has a name {stats['pos_named'] / max(1, n):.0%}, ops { {k: v for k, v in stats.items() if k.startswith('op_')} }", flush=True)


if __name__ == "__main__":
    main()

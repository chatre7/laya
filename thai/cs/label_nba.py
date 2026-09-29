"""Next-best-action data (banking) for laya, run nba1.

The teacher judges the MESSAGE only: it answers NBA_Q for a verified, first-contact customer. The playbook rule
(nba_actions.apply_context) then carries that distribution into the other three contexts (not verified / third contact), so
one teacher call gives four training states and the context policy is taught exactly as written, not as the teacher guesses it.

Texts: every Pantip banking post except the 120 hand-labelled eval rows, plus synthetic banking messages from cs9. Replay:
samples of the cs9 and Pantip items, so intent / department / urgency are not forgotten.
Eval: the 120 real banking rows x 4 contexts, gold from data_domain/nba_labels_banking_eval.json + apply_context.

    python label_nba.py --teacher http://172.18.72.145:8010 --workers 8
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from distill_from_ots import ask_teacher, targets_from  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402
from nba_actions import ACTIONS, CONTEXTS, NBA_BASE_Q, NBA_Q, apply_context, state_of  # noqa: E402

KEYS = list(ACTIONS)


def key_of(text):
    return " ".join(text.split())[:60]


def push(probs, verified, contacts):
    """Teacher distribution for the base context -> distribution in another context, through the playbook rule."""
    out = [0.0] * len(KEYS)
    for a, p in zip(KEYS, probs):
        out[KEYS.index(apply_context(a, verified, contacts))] += p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="/work/thai/data/domain")
    ap.add_argument("--cs", default="/work/thai/data/cs")
    ap.add_argument("--labels", default="/work/thai/data_domain/nba_labels_banking_eval.json")
    ap.add_argument("--out", default="/work/thai/data/nba")
    ap.add_argument("--prefix", default="nba1")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synthetic", type=int, default=5000, help="synthetic banking messages from cs9")
    ap.add_argument("--replay-cs", type=int, default=12000)
    ap.add_argument("--replay-pantip", type=int, default=6000)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--student", default="/work/thai/out/laya-th-run8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: texts to label")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---- eval: 120 real banking rows x 4 contexts
    rows = [json.loads(l) for l in open(Path(args.domain) / "real_cs_eval.jsonl", encoding="utf-8")]
    bank = [r for r in rows if r["source"] == "banking"]
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]
    held_out = {key_of(r["state"]) for r in bank}
    with open(out / "nba_eval_banking.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(bank):
            acc = labels[str(i)].split("|")
            for v, h in CONTEXTS:
                f.write(json.dumps({"id": f"{r['id']}-v{int(v)}-h{h}", "source": f"ctx_v{int(v)}_h{h}", "state": state_of(r["state"], v, h),
                                    "questions": {"next_action": NBA_Q}, "labels": {"next_action": apply_context(acc[0], v, h)},
                                    "accept": sorted({apply_context(a, v, h) for a in acc}), "intent": r["labels"]["intent"]},
                                   ensure_ascii=False) + "\n")
    print(f"eval: {len(bank)} rows x {len(CONTEXTS)} contexts; gold actions {dict(Counter(l.split('|')[0] for l in labels.values()))}", flush=True)

    # ---- texts to label
    texts = []
    for line in open(Path(args.domain) / "pantip_banking.jsonl", encoding="utf-8"):
        t = json.loads(line)["text"].strip()
        if len(t) >= 10 and key_of(t) not in held_out:
            texts.append(("pantip", t))
    syn = [json.loads(l) for l in open(Path(args.cs) / "cs9.jsonl", encoding="utf-8")]
    syn = [r["state"] for r in syn if r.get("labels", {}).get("business") == "banking" and isinstance(r["state"], str)
           and not r["source"].startswith("pantip")]
    rng.shuffle(syn)
    texts += [("synthetic", t) for t in syn[: args.synthetic]]
    if args.limit:
        rng.shuffle(texts)
        texts = texts[: args.limit]
    print(f"{len(texts)} texts to label: {dict(Counter(s for s, _ in texts))}", flush=True)

    done = [0]

    def work(st):
        src, text = st
        res = ask_teacher(args.teacher, text, {"next_action": NBA_BASE_Q})
        done[0] += 1
        if done[0] % 1000 == 0:
            el = time.perf_counter() - t0
            print(f"  {done[0]}/{len(texts)} labelled, {el / 60:.1f} min", flush=True)
        if res is None:
            return None
        base = dict(zip(NBA_BASE_Q["criteria"], targets_from(res["answers"], {"next_action": NBA_BASE_Q})["next_action"]))
        return {"source": src, "text": text, "probs": [base.get(k, 0.0) for k in KEYS]}  # verify_identity: 0 in the base context

    with ThreadPoolExecutor(args.workers) as ex:
        labelled = [x for x in ex.map(work, texts) if x]
    with open(out / f"{args.prefix}_teacher.jsonl", "w", encoding="utf-8") as f:
        for r in labelled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    top = Counter(KEYS[max(range(len(KEYS)), key=r["probs"].__getitem__)] for r in labelled)
    print(f"teacher labelled {len(labelled)}/{len(texts)} in {(time.perf_counter() - t0) / 60:.1f} min; teacher top-1 {dict(top.most_common())}", flush=True)

    # ---- items: every labelled text in all four contexts + replay
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    qi = {"t": "choice", "ins": NBA_Q["instructions"], "crit": NBA_Q["criteria"]}
    items, dropped = [], 0
    for r in labelled:
        for v, h in CONTEXTS:
            seq, markers = build_sequence(tok, state_of(r["text"], v, h), qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            t = [(1 - args.smooth) * p + args.smooth / len(KEYS) for p in push(r["probs"], v, h)]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": t,
                          "label": max(range(len(t)), key=t.__getitem__), "source": f"nba_{r['source']}"})
    n_nba = len(items)
    for name, n in (("cs9_items.pt", args.replay_cs), ("pantip9_items.pt", args.replay_pantip)):
        pool = torch.load(Path(args.cs) / name, weights_only=True)  # our own item lists: plain lists/dicts/numbers
        items += rng.sample(pool, min(n, len(pool)))
        del pool
    rng.shuffle(items)
    torch.save(items, out / f"{args.prefix}_items.pt")
    summary = {"texts": len(texts), "labelled": len(labelled), "nba_items": n_nba, "dropped": dropped, "items": len(items),
               "replay_cs": args.replay_cs, "replay_pantip": args.replay_pantip, "teacher_top1": dict(top), "smooth": args.smooth,
               "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items)), "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / f"{args.prefix}_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

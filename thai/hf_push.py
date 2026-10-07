"""Back up served checkpoints to a private Hugging Face repo, one folder per run, with a short card. Needs a write token
on the box (`~/.cache/huggingface/token`, see hf_push.sh). `checkpoint_latest` (the trainer's resume state) is left out.

    python hf_push.py --repo Chatre7/laya-th --runs 16,19
"""
import argparse
import os

from huggingface_hub import HfApi

NOTES = {
    "16": "Run 16 (2026-10-03): run 14 + composed sarcasm sets, star-labelled reviews and Wisesight neutral / question. Served 2026-10-03 to 10-07. "
          "Real posts intent / department / urgency 0.614 / 0.722 / 0.811; reviews 0.723 / 0.747 / 0.840; stars agreement 0.861; sarcasm set 43/100.",
    "19": "Run 19 (2026-10-07): run 16 + two new questions taught by Qwen3-8B with written rules - churn threat (39 of 51 right, 39 of 49 found on the "
          "hand-checked set, equal to the teacher) and contact effort (81% exact level). Triage unchanged. Served since 2026-10-07.",
}
CARD = """---
license: other
language: [th]
library_name: laya
tags: [laya, thai, call-center, decision-model]
---
# laya-th: Thai call-center decision model (private backup)

Fine-tunes of convaiinnovations/laya for Thai customer-service triage (telecom / banking / insurance): intent, department,
urgency, frustration, sentiment, and since run 19 a churn threat and contact effort. One folder per run; load with
`laya.Agent("<local copy of a run folder>")`. Training data includes forum posts and app reviews written by others and the
answers of teacher models; this repo is private and not for redistribution. Code and results: github.com/chatre7/laya, branch thai.

{rows}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Chatre7/laya-th")
    ap.add_argument("--runs", default="16,19")
    ap.add_argument("--out", default="/work/thai/out")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(args.repo, private=True, exist_ok=True)
    runs = args.runs.split(",")
    for r in runs:
        folder = os.path.join(args.out, f"laya-th-run{r}")
        print(f"uploading {folder} -> {args.repo}/run{r}", flush=True)
        api.upload_folder(repo_id=args.repo, folder_path=folder, path_in_repo=f"run{r}", ignore_patterns=["checkpoint_latest/*", "checkpoint_latest"],
                          commit_message=f"laya-th run {r}")
    rows = "\n".join(f"- `run{r}/`: {NOTES.get(r, '')}" for r in runs)
    api.upload_file(path_or_fileobj=CARD.format(rows=rows).encode("utf-8"), path_in_repo="README.md", repo_id=args.repo, commit_message="card")
    print("done:", api.repo_info(args.repo).id, "private" if api.repo_info(args.repo).private else "PUBLIC")


if __name__ == "__main__":
    main()

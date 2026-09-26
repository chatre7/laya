"""Fetch English customer-service sets made of REAL user questions (not templates): banking77 (77 fine banking intents,
CC-BY-4.0, ~13k) and insurance-qa-en (real insurance questions with a topic, ~16k). Written as JSONL with the same fields
translate_colloquial.py expects (text_en, business, category, intent), so the EN -> Thai colloquial step can run on them.

    python fetch_real_en.py --out /work/thai/data/cs2
"""
import argparse
import collections
import json
from pathlib import Path

from datasets import load_dataset

INSURANCE_TOPIC_TO_INTENT = {  # insurance-qa topics -> our insurance intents (all "information_*"), others -> general_information
    "life-insurance": "information_life_insurance", "auto-insurance": "information_auto_insurance", "health-insurance": "information_health_insurance",
    "home-insurance": "information_home_insurance", "homeowners-insurance": "information_home_insurance", "renters-insurance": "information_home_insurance",
    "travel-insurance": "information_travel_insurance", "pet-insurance": "information_pet_insurance",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data/cs2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("mteb/banking77", split="train")
    rows = [{"text_en": r["text"], "business": "banking", "category": "BANKING77", "intent": r["label_text"]} for r in ds]
    cnt = collections.Counter(r["intent"] for r in rows)
    with open(out / "banking77_en.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"banking77: {len(rows)} rows, {len(cnt)} intents; smallest {cnt.most_common()[-3:]}")
    print("   intents:", ", ".join(sorted(cnt)))

    ds = load_dataset("rvpierre/insurance-qa-en", split="train")
    rows, topics = [], collections.Counter()
    for r in ds:
        q = " ".join(str(r["question_en"]).split())
        topic = str(r["topic_en"]).strip()
        topics[topic] += 1
        rows.append({"text_en": q, "business": "insurance", "category": topic.upper(), "intent": INSURANCE_TOPIC_TO_INTENT.get(topic, "general_information")})
    with open(out / "insuranceqa_en.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"insurance-qa: {len(rows)} rows; topics {topics.most_common(12)}")


if __name__ == "__main__":
    main()

"""English customer-service datasets with REAL user questions (not templates) that could be translated: check licence/size."""
from huggingface_hub import HfApi

api = HfApi()
cands = ["PolyAI/banking77", "clinc/clinc_oos", "PolyAI/nlu_plus_plus", "pjaol/nluplusplus", "deepset/insurance_qa", "fvillena/insurance_qa",
         "InsuranceQA/insuranceQA", "AmazonScience/massive", "bitext/Bitext-mortgage-and-loans-llm-chatbot-training-dataset",
         "bitext/Bitext-wealth-management-llm-chatbot-training-dataset", "bitext/Bitext-media-llm-chatbot-training-dataset",
         "bitext/Bitext-travel-llm-chatbot-training-dataset", "bitext/Bitext-hospitality-llm-chatbot-training-dataset"]
for c in cands:
    try:
        d = api.dataset_info(c)
        tags = [t for t in (d.tags or []) if t.startswith(("license:", "size_categories:"))]
        print(f"  OK {d.downloads or 0:>7} {c:70s} {' '.join(tags)}")
    except Exception:  # noqa: BLE001
        print(f"  -- {c}")
print("--- search: real customer questions ---")
seen = {}
for q in ["banking77", "insurance qa", "insurance questions", "telecom intent", "telco intent", "customer support intent", "bank intent",
          "customer service intent dataset", "support tickets", "twitter customer support", "helpdesk tickets", "nlu++", "hwu64", "atis banking"]:
    for d in api.list_datasets(search=q, limit=15):
        seen.setdefault(d.id, d)
rows = sorted(((d.downloads or 0, d.id, [t for t in (d.tags or []) if t.startswith(("license:", "size_categories:", "language:"))]) for d in seen.values()), reverse=True)
for dl, i, t in rows[:45]:
    print(f"  {dl:>7} {i:70s} {' '.join(t)[:70]}")

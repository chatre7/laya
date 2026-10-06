"""Hand labels for the two new-question check samples, by line number of their print files; every text not listed is
no / no / no / level 0. Written 2026-10-06 without any model's answers in view.
  sample 1 (new_check_sample.jsonl, 127)   drawn by the first teacher's answers (sample_new_check.py)
  sample 2 (new_check_sample2.jsonl, 180)  drawn by wording, 45 texts with a cue word per question (sample_new_check2.py)
Rules used (second version, after reading the first labeller's misses: the first version made "leaving" depend on which
company is being asked, which the text does not say, and counted a public warning post as an outside threat):
  churn_threat     the writer talks about leaving their CURRENT provider, bank or insurer: will, wants to, is thinking of
                   or has already cancelled the service, closed the account, not renewed, or ported away - also when they
                   ask for a port-in deal or for what other providers offer. Not: cancelling one package or changing plan
                   with the same company, reinstalling the app, telling others to leave.
  external_threat  the writer will take the company to an outside body: a regulator or consumer body, the police or a
                   court, or says they will file a complaint about the company. Not: a police report about a scammer, being
                   sued by the company, staff forwarding something to a regulator, news about other people, an angry review.
  third_party      the matter concerns someone else's account, line, policy or property: a parent's SIM, a dead relative's
                   account, cover bought for a child, asking for one's mother. Using one's own product is not.
  contact_effort   contact with the company about this matter before now, including an attempt that did not get through:
                   0 none mentioned, 1 once (called, went to the branch, wrote, was told something by staff), 2 a few
                   times, 3 many times or several channels and still open. Retrying the app is not contact.

    python make_new_labels.py      # -> data_domain/new_labels.json  {id: [churn, external, third_party, effort]}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, "..", "data_domain")
SAMPLES = {
    "new_check_sample.jsonl": (
        {0, 9, 40, 43, 82, 117},
        set(),
        {3, 20, 47, 53, 59, 108},
        {14: 1, 22: 1, 42: 2, 44: 1, 52: 1, 56: 2, 60: 1, 61: 1, 74: 3, 90: 1, 97: 1, 103: 3, 114: 1, 115: 1}),
    "new_check_sample2.jsonl": (
        {0, 8, 9, 11, 14, 17, 24, 29, 41, 45, 46, 50, 52, 56, 59, 65, 76, 78, 84, 86, 95, 107, 110, 117, 120, 121, 122, 123, 124, 125, 129, 131, 136,
         143, 149, 153, 156, 162, 163, 164, 166, 174, 178},
        {18, 53, 54, 87, 136, 148, 159},
        {1, 23, 30, 33, 39, 49, 58, 79, 84, 116, 123, 129, 144, 151},
        {1: 1, 6: 1, 7: 1, 9: 1, 13: 3, 14: 1, 16: 1, 18: 1, 22: 1, 24: 1, 31: 3, 37: 1, 41: 3, 42: 1, 50: 1, 51: 1, 55: 1, 60: 1, 64: 1, 79: 1, 80: 1,
         86: 1, 87: 1, 91: 1, 93: 3, 103: 1, 104: 1, 111: 2, 114: 1, 121: 1, 127: 1, 128: 1, 130: 1, 131: 1, 133: 1, 134: 3, 136: 3, 146: 3, 157: 3,
         167: 1, 171: 1, 176: 3, 177: 1}),
}

labels, rows_all = {}, []
for name, (churn, external, third, effort) in SAMPLES.items():
    rows = [json.loads(l) for l in open(os.path.join(DD, name), encoding="utf-8")]
    for i, r in enumerate(rows):
        labels[r["id"]] = [int(i in churn), int(i in external), int(i in third), effort.get(i, 0)]
        rows_all.append({"id": r["id"], "source": r["source"], "business": r["business"], "text": r["text"], "sample": 1 if name == "new_check_sample.jsonl" else 2})
json.dump({"questions": ["churn_threat", "external_threat", "third_party", "contact_effort"], "labels": labels},
          open(os.path.join(DD, "new_labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
with open(os.path.join(DD, "new_check_all.jsonl"), "w", encoding="utf-8") as f:
    for r in rows_all:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(labels), "texts; yes per question", [sum(v[k] > 0 for v in labels.values()) for k in range(4)])

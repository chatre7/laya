"""The new questions (cs_questions.EXTRA) answered by a generative LLM with written rules instead of a decision model.
Clef-Flash read any angry review as a threat to leave (5 of 60 "yes" right, eval_new.py); these four questions are about
whether something is *said*, which a model that reads the rules and a few examples can check. Qwen3 on vLLM (vllm_up.sh),
schema-constrained JSON, one call per text for all four answers. Writes the same file format as label_new_llm.py, with
hard answers (yes = 1.0, the level = 1.0), so eval_new.py and the item builder read either.

    python3 thai/cs/label_new_gen.py --eval                                   # the hand-labelled texts only
    python3 thai/cs/label_new_gen.py --out thai/data/cs/new_gen.jsonl         # the whole pool; resumes
    python3 thai/cs/label_new_gen.py --cue-only --url http://localhost:8013 --model saluki --out thai/data/cs/new_saluki.jsonl --workers 4
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_new_llm  # noqa: E402

RULES = """คุณเป็นหัวหน้าทีมบริการลูกค้าของธุรกิจมือถือ ธนาคาร และประกัน อ่านข้อความของลูกค้าแล้วตอบ 4 ข้อเป็น JSON ตอบตามสิ่งที่ข้อความ "พูดไว้จริง" เท่านั้น ห้ามเดาจากอารมณ์ ข้อความที่บ่น ด่า หรือโกรธมากแต่ไม่ได้พูดเรื่องนั้น ต้องตอบ false หรือ 0

1. churn_threat (true/false): ผู้เขียนพูดถึงการเลิกใช้หรือย้ายออกจากผู้ให้บริการ ธนาคาร หรือบริษัทประกัน "ที่ตัวเองใช้อยู่ตอนนี้" หรือไม่
   true = บอกว่าจะ อยากจะ กำลังคิดจะ หรือได้เลิกใช้ ปิดบัญชี ไม่ต่ออายุ ย้ายค่าย หรือย้ายไปเจ้าอื่นแล้ว, ถามหาโปรย้ายค่าย, ถามว่าเจ้าอื่นมีโปรอะไรเพื่อจะย้าย, บอกว่าถ้าไม่แก้จะเสียลูกค้า (ไม่ว่ากำลังถามเจ้าเดิมหรือเจ้าใหม่ก็ถือว่า true)
   false = ไม่ได้พูดถึงการเลิกหรือย้ายของตัวเอง, แค่ขอยกเลิกแพ็กเกจเสริมหรือเปลี่ยนโปรกับเจ้าเดิม, ลบแอปแล้วติดตั้งใหม่, บอกให้คนอื่นเลิกใช้
2. external_threat (true/false): ผู้เขียนบอกว่าจะร้องเรียน ฟ้อง หรือแจ้งความเพื่อเอาผิดบริษัทหรือไม่
   true = จะหรือได้ร้องเรียน สคบ. กสทช. ธปท. คปภ. หรือหน่วยงานรัฐ, จะฟ้องหรือแจ้งความเอาผิดบริษัท, บอกว่าจะร้องเรียนบริษัท หรือถามว่าจะร้องเรียนบริษัทได้ที่ไหน
   false = แจ้งความเรื่องมิจฉาชีพหรือคนโกง (ไม่ใช่เอาผิดบริษัท), กลัวว่าตัวเองจะถูกฟ้อง, พนักงานบอกว่าจะส่งเรื่องให้หน่วยงาน, ข่าวหรือเรื่องของคนอื่น, แค่เอ่ยชื่อหน่วยงาน, รีวิวด่าแรงแต่ไม่ได้บอกว่าจะร้องเรียน
3. third_party (true/false): เรื่องนี้เป็นของบัญชี เบอร์ กรมธรรม์ หรือทรัพย์สินของคนอื่นที่ไม่ใช่ผู้เขียนหรือไม่
   true = ของพ่อแม่ ญาติ ลูก คู่สมรส หรือผู้เสียชีวิต, ถามแทนคนอื่น, ทำเรื่องหรือซื้อประกันให้คนอื่น
   false = เป็นของผู้เขียนเอง หรือไม่ได้บอก, แค่เอ่ยถึงคนในครอบครัวโดยที่เรื่องเป็นของตัวเอง
4. contact_effort (0-3): ก่อนหน้านี้ผู้เขียนเคยติดต่อหรือพยายามติดต่อบริษัทเรื่องนี้มาแล้วมากแค่ไหน (โทร ไปสาขา/ศูนย์ แชต ทักเพจ แจ้งเรื่อง หรือคุยกับพนักงาน/ตัวแทน)
   0 = ไม่ได้บอกว่าเคยติดต่อ (ลองเข้าแอปซ้ำ ลงทะเบียนซ้ำ ไม่นับเป็นการติดต่อ)
   1 = เคยติดต่อ ได้คุยกับพนักงาน หรือพยายามติดต่อแล้วหนึ่งครั้ง (โทรแล้วไม่มีคนรับ ติดต่อไม่ได้ ก็นับ)
   2 = ติดต่อซ้ำสองสามครั้ง
   3 = ติดต่อหลายครั้งหรือหลายช่องทางแล้วเรื่องยังไม่จบ

ตัวอย่าง:
"แอปห่วยมาก เข้าไม่ได้เลย ปรับปรุงด่วน" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 0}
"เน็ตหลุดทั้งวัน โทรแจ้งสามรอบแล้วไม่มีใครมา ถ้ายังเป็นแบบนี้สิ้นเดือนย้ายค่ายแน่" -> {"churn_threat": true, "external_threat": false, "third_party": false, "contact_effort": 2}
"ใช้ดีแทคอยู่ครับ อยากย้ายไปเอไอเอส มีโปรย้ายค่ายเบอร์เดิมอะไรบ้าง" -> {"churn_threat": true, "external_threat": false, "third_party": false, "contact_effort": 0}
"ตอนนี้ใช้ทรูเดือนละ 599 ค่ายไหนมีโปรเน็ตไม่อั้นถูกกว่านี้บ้างครับ" -> {"churn_threat": true, "external_threat": false, "third_party": false, "contact_effort": 0}
"อยากยกเลิกแพ็กเกจเสริมดูหนัง 99 บาท ทำยังไงคะ" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 0}
"ขอเปลี่ยนโปรจาก 399 เป็น 299 ของค่ายเดิมได้ไหมคะ" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 0}
"เคลมมาสองเดือนแล้วเงียบ ทั้งโทรทั้งไลน์ทั้งไปสาขา พรุ่งนี้จะไปร้อง คปภ." -> {"churn_threat": false, "external_threat": true, "third_party": false, "contact_effort": 3}
"บริการแบบนี้จะร้องเรียนบริษัทให้ถึงที่สุด" -> {"churn_threat": false, "external_threat": true, "third_party": false, "contact_effort": 0}
"โดนมิจฉาชีพหลอกโอนเงิน แจ้งความแล้ว ธนาคารช่วยอายัดบัญชีปลายทางได้ไหม" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 0}
"ข่าว: ลูกค้ารายหนึ่งยื่นฟ้องบริษัทประกัน เรียกค่าเสียหาย 5 ล้านบาท" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 0}
"พนักงานสาขาบอกว่าจะส่งเรื่องให้แบงก์ชาติให้ แต่ไม่ให้เอกสารอะไรเลย" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 1}
"พ่อเสียแล้ว อยากปิดบัญชีของพ่อ ต้องใช้เอกสารอะไรบ้าง" -> {"churn_threat": false, "external_threat": false, "third_party": true, "contact_effort": 0}
"โทรไปคอลเซ็นเตอร์ก็ไม่มีคนรับสาย" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 1}
"ไปสาขามาเมื่อวาน พนักงานบอกให้รอ 7 วัน" -> {"churn_threat": false, "external_threat": false, "third_party": false, "contact_effort": 1}
"""
SCHEMA = {"type": "object", "properties": {"churn_threat": {"type": "boolean"}, "external_threat": {"type": "boolean"}, "third_party": {"type": "boolean"},
                                           "contact_effort": {"type": "integer", "enum": [0, 1, 2, 3]}},
          "required": ["churn_threat", "external_threat", "third_party", "contact_effort"], "additionalProperties": False}


def ask(url, model, text, max_chars=1500):
    body = {"model": model, "messages": [{"role": "user", "content": RULES + "\nข้อความลูกค้า: " + text[:max_chars]}], "temperature": 0, "max_tokens": 60,
            "response_format": {"type": "json_schema", "json_schema": {"name": "label", "schema": SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                o = json.loads(json.load(r)["choices"][0]["message"]["content"])
            lv = int(o["contact_effort"])
            return {"churn_threat": float(bool(o["churn_threat"])), "external_threat": float(bool(o["external_threat"])),
                    "third_party": float(bool(o["third_party"])), "contact_effort": [float(k == lv) for k in range(4)]}
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--root", default="thai")
    ap.add_argument("--out", default="thai/data/cs/new_gen.jsonl")
    ap.add_argument("--eval", action="store_true", help="label only the hand-checked texts -> <out>.eval.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cue-only", action="store_true", help="only pool texts that contain a cue word of any question (sample_new_check2.CUES)")
    args = ap.parse_args()
    if args.eval:
        rows = [json.loads(l) for l in open(f"{args.root}/data_domain/new_check_all.jsonl", encoding="utf-8")]
        out = args.out.replace(".jsonl", ".eval.jsonl")
        if os.path.exists(out):
            os.remove(out)
    else:
        rows = label_new_llm.pool(args.root)
        if args.cue_only:  # a slow teacher goes only where a "yes" can be: ~4,000 of the 27,716 texts
            import re
            from sample_new_check2 import CUES
            rx = re.compile("|".join(f"(?:{p})" for p in CUES.values()), re.I)
            rows = [r for r in rows if rx.search(r["text"])]
        out = args.out
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()} if os.path.exists(out) else set()
    todo = [r for r in rows if r["id"] not in done][: args.limit or None]
    print(f"{len(rows)} texts {dict(Counter(r['source'] for r in rows))}; {len(done)} already labelled, {len(todo)} to do", flush=True)
    t0, n, failed = time.time(), 0, 0
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for r, a in zip(todo, ex.map(lambda r: ask(args.url, args.model, r["text"]), todo)):
            n += 1
            if a is None:
                failed += 1
                continue
            f.write(json.dumps({"id": r["id"], "source": r["source"], "business": r["business"], "text": r["text"][:1500], "answers": a}, ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                f.flush()
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.1f} min  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print(f"labelled {n - failed}/{len(todo)} in {(time.time() - t0) / 60:.1f} min ({failed} failed) -> {out}", flush=True)


if __name__ == "__main__":
    main()

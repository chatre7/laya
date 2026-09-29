"""Translate the unique Mind2Web task instructions to Thai with the local vLLM (Qwen3-4B), the way a Thai user would type the
request to an assistant. Names, numbers, dates, emails, site and button names stay as they are.

    python3 thai/web/translate_tasks.py --data thai/data/web --out thai/data/web/tasks_th.json
"""
import argparse
import glob
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("แปลคำสั่งต่อไปนี้เป็นภาษาไทย ให้เป็นประโยคที่คนไทยพิมพ์สั่งผู้ช่วยให้ทำงานบนเว็บไซต์ "
          "คงชื่อคน ชื่อสถานที่ ชื่อเว็บไซต์ ชื่อสินค้า ตัวเลข วันที่ เวลา อีเมล และรหัสไปรษณีย์ไว้ตามต้นฉบับ "
          "ตอบเฉพาะคำแปลบรรทัดเดียว ไม่ต้องอธิบาย\n\nคำสั่ง: {task}")


def chat(url, model, content, timeout=120):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.3, "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
                return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"“”').splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return ""


def ok(t):
    thai = sum(1 for ch in t if "฀" <= ch <= "๿")
    return len(t) >= 6 and thai >= 5 and not re.search(r"(.)\1{6,}", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="thai/data/web")
    ap.add_argument("--out", default="thai/data/web/tasks_th.json")
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    tasks = sorted({json.loads(l)["task_en"] for f in glob.glob(f"{args.data}/m2w_*.jsonl") for l in open(f, encoding="utf-8")})
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda t: chat(args.url, args.model, PROMPT.format(task=t)), tasks))
    res = {t: o for t, o in zip(tasks, outs) if ok(o)}
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"translated {len(res)}/{len(tasks)} tasks in {(time.time() - t0) / 60:.1f} min")
    for t in tasks[:5]:
        print(" ", t[:100], "\n   ->", res.get(t, "(failed)")[:120])


if __name__ == "__main__":
    main()

"""Questions neither our items nor laya-thai-decisions contain, on messages written for this check: does a checkpoint
answer a question it was never trained on? Prints p(yes) (or the chosen level) per model.

    python new_questions_demo.py --models /work/thai/out/laya-th-run16,/work/thai/out/laya-th-run18
"""
import argparse

from agents import load_agent

CASES = [  # (message, question id, expected)
    ("เน็ตหลุดทุกวัน แจ้งไปสามรอบแล้วยังไม่มีใครมาดู ถ้าสิ้นเดือนนี้ยังไม่ได้เรื่องจะย้ายค่ายแล้วนะ", "churn", True),
    ("ขอสอบถามโปรเน็ตรายเดือนที่ถูกที่สุดตอนนี้หน่อยครับ", "churn", False),
    ("เน็ตหลุดทุกวัน แจ้งไปสามรอบแล้วยังไม่มีใครมาดู ถ้าสิ้นเดือนนี้ยังไม่ได้เรื่องจะย้ายค่ายแล้วนะ", "not_churn", False),
    ("ขอสอบถามโปรเน็ตรายเดือนที่ถูกที่สุดตอนนี้หน่อยครับ", "not_churn", True),
    ("มีคนโทรมาบอกว่าเป็นเจ้าหน้าที่ธนาคาร ให้กดลิงก์แล้วกรอกรหัส OTP ตอนนี้เงินหายไปสองหมื่น", "scam", True),
    ("อยากทราบว่าบัตรเดบิตใบใหม่ใช้กดเงินต่างประเทศได้ไหมคะ", "scam", False),
    ("จะไปร้องเรียน สคบ. กับลงโซเชียลให้หมด ถ้าไม่คืนเงินภายในวันนี้", "threat", True),
    ("รบกวนช่วยตรวจสอบยอดให้หน่อยนะคะ ขอบคุณค่ะ", "threat", False),
    ("แม่ผมอายุ 78 ใช้แอปไม่เป็น ผมเลยทำรายการแทนให้ครับ แต่ระบบให้สแกนหน้า", "third_party", True),
    ("ผมลืมรหัสผ่านแอปครับ ต้องทำยังไง", "third_party", False),
    ("ขอบคุณน้องพนักงานที่ช่วยแก้ปัญหาให้เร็วมากครับ ประทับใจ", "effort", 0),
    ("โทรไปห้ารอบ รอสายรอบละครึ่งชั่วโมง ไปสาขาอีกสองครั้งก็ยังไม่จบ", "effort", 3),
]
QUESTIONS = {
    "churn": {"type": "noul", "instructions": "ลูกค้าขู่ว่าจะยกเลิกหรือย้ายไปใช้เจ้าอื่นหรือไม่"},
    "not_churn": {"type": "noul", "instructions": "ลูกค้าไม่ได้พูดถึงการยกเลิกหรือย้ายไปใช้เจ้าอื่นเลยใช่หรือไม่"},
    "scam": {"type": "noul", "instructions": "ข้อความนี้เกี่ยวกับการถูกหลอกหรือมิจฉาชีพหรือไม่"},
    "threat": {"type": "noul", "instructions": "ลูกค้าขู่ว่าจะร้องเรียนหน่วยงานภายนอกหรือเผยแพร่เรื่องนี้หรือไม่"},
    "third_party": {"type": "noul", "instructions": "ผู้เขียนติดต่อมาแทนคนอื่นหรือไม่"},
    "effort": {"type": "score", "instructions": "ลูกค้าต้องออกแรงติดต่อมากแค่ไหนกว่าจะถึงตอนนี้",
               "criteria": ["ไม่ต้องออกแรงเลย", "ติดต่อครั้งเดียว", "ติดต่อซ้ำสองสามครั้ง", "ติดต่อหลายครั้งหลายช่องทาง"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for path in args.models.split(","):
        agent = load_agent(path, args.device)
        ok = 0
        print("==", path.rstrip("/").split("/")[-1])
        for text, qid, exp in CASES:
            a = agent.predict(text, {qid: QUESTIONS[qid]})["answers"][qid]
            if QUESTIONS[qid]["type"] == "noul":
                got, shown = a["noul"] > 0.5, f"p(yes) {a['noul']:.2f}"
            else:
                got = max(range(4), key=lambda i: a["probabilities"][str(i)])
                shown = f"level {got}"
            ok += got == exp
            print(f"  {'ok ' if got == exp else 'BAD'} {qid:11s} want {str(exp):5s} {shown:12s} | {text[:60]}")
        print(f"  {ok}/{len(CASES)}")
        del agent


if __name__ == "__main__":
    main()

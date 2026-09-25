# Proposed intent set for the real call-center data (draft, 2026-09-25)

From reading `System One/data/customer_messages.csv` (200 real-style messages) and `Qwen_csv_*.txt` (154, refund-heavy): the
messages are **e-commerce / marketplace support**, and only about half of them map onto Bitext's 27 intents. On these texts
the two models agree on intent only 39% of the time, and run 5 answers `other` for 104/200, i.e. "not one of the 27": returns,
damaged goods, warranty, loyalty points, price drops, stock and product questions have no home in the Bitext list.

Proposed replacement (edit freely; the descriptions are what the model reads, so keep them concrete):

| id | Thai description (criteria text) | seen in the data |
|---|---|---|
| `track_order` | ของยังไม่ถึง ถามสถานะพัสดุ tracking ผิดปกติ ส่งช้า | very common |
| `cancel_order` | ขอยกเลิกออเดอร์ สั่งผิด เปลี่ยนใจ ระบบไม่ให้ยกเลิก | common |
| `change_order` | แก้จำนวน เปลี่ยนรุ่น/สี/ไซส์ ก่อนส่ง | common |
| `change_shipping` | เปลี่ยนที่อยู่จัดส่ง เลื่อนวันส่ง ขอส่งด่วน ส่งแยกหลายรอบ | common |
| `return_item` | ขอคืนสินค้า ถามเงื่อนไขการคืน ไซส์ไม่พอดี ไม่ถูกใจ | common |
| `damaged_or_wrong_item` | ของชำรุด กล่องบุบ ของหมดอายุ ได้ของไม่ตรง ได้ไม่ครบ | common |
| `refund_status` | ถามว่าเงินคืนถึงไหน คืนช้า คืนผิดบัญชี ขอสลิป | very common in file 2 |
| `refund_request` | ขอเงินคืน ขอส่วนต่าง ยกเลิกแล้วยังโดนหัก | common |
| `payment_issue` | จ่ายไม่ผ่าน ตัดเงินแต่ไม่มีออเดอร์ ผ่อนชำระ ใบเสร็จ/ใบกำกับภาษี | some |
| `promotion_or_price` | ถามโปรโมชั่น โค้ดส่วนลด ราคาลดหลังสั่ง ของแถมไม่มา | some |
| `product_question` | ถามสต็อก สี ไซส์ วิธีใช้ สเปก ประกัน ศูนย์ซ่อม | common |
| `account_issue` | ล็อกอินไม่ได้ ลืมรหัส บัญชีถูกระงับ แก้ข้อมูล/ที่อยู่ในบัญชี | some |
| `loyalty_points` | คะแนนสะสม แลกของไม่ได้ คูปอง | few |
| `app_or_website_bug` | แอป/เว็บล่ม กดไม่ได้ หน้าค้าง | some |
| `complaint_service` | ร้องเรียนบริการ แชทตอบช้า พนักงานไม่สุภาพ คุณภาพต่ำกว่าโฆษณา | common |
| `contact_human` | ขอคุยกับพนักงาน ขอเบอร์/ช่องทางติดต่อ | few |
| `praise` | ชม ขอบคุณ | few |
| `other` | ไม่เข้าข่ายข้อใดข้างต้น ไม่ใช่เรื่องบริการลูกค้า ล้อเล่น ไม่มีความหมาย | ~10% (ดาวอังคาร, พลังวิเศษ, แมว, พิซซ่า) |

Keep `category` / `department` / `urgency` / `frustration` / `wants_refund` / `wants_human` / `has_order_ref` / `sentiment`
as they are; they transferred fine (agreement 0.88-1.00 on the real texts, urgency 0.73-0.89 is the one to review).

How to adopt it:

1. Edit the list above into `cc_questions.py` (`INTENTS_ECOM`, and point `QUESTIONS["intent"]["criteria"]` at it).
2. Review the two sheets with the new list (re-run `draft_labels.py` so the dropdowns and drafts use it).
3. Re-label the 69k training texts with the new intent question (`label_cc.py`, teacher-only for intent since the Bitext
   human labels no longer apply, ~1 h with two teachers) and train run 6 from run 3. The teacher is weak on these intents
   too (it called 61/200 messages `complaint`), so the reviewed real sheets should also go into training as human items,
   with a held-out slice kept for the score.

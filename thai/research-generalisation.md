# Making the Thai laya student generalise: what the sources say (2026-09-23)

Question: runs 1-3 of the Thai student (mmBERT-base, 322M, distilled from OpenThai-SystemOne) gain a lot in-domain but the
held-out sets stay flat (wisesight 0.587 -> 0.557, sib200_th 0.745 -> 0.701 from run 2 to run 3; teacher 0.547 / 0.784).
What, according to primary sources, would move held-out accuracy, what would not, and what does each cost on one A2 (16 GB)?

Sources are cited inline as URL or `path:line`. "Ours" = this directory (`thai/README.md`, scripts, `results/`).

## (a) Diagnosis: why held-out is flat

1. **The student is bounded by the teacher on exactly the sets that are flat.** On wisesight the teacher itself scores
   0.547 on our slice (`thai/README.md`, run tables) and 51.6 on the full set, which its authors call the "weakest Thai set"
   (https://huggingface.co/iapp/OpenThai-SystemOne, held-out table). Run 2 already reached 0.587 there, above the teacher.
   Distillation transfers the teacher's function (Hinton et al.: the student learns from "soft targets produced by the teacher
   network", https://arxiv.org/pdf/1503.02531), so more teacher labels cannot push wisesight past ~0.55-0.59. The flat line
   is the ceiling, not a training failure. sib200 (teacher 0.784, student 0.70-0.75) still has ~5-8 points of teacher headroom.

2. **Run 3 traded held-out for in-domain by mixing human labels back in.** Run 3 added the 10,674 human-labelled sequences of
   run 1 (`thai/run3.sh:20`) whose sources are the in-domain sets; those rose 5-30 points while sib200 fell 4.4 points
   (`results/run3.json`). Run 1 alone showed the same pattern: large in-domain gains, held-out flat or down
   (`thai/README.md`, "Run 1"). Human labels from five narrow sources pull the model toward those label distributions.

3. **The transfer set is small and narrow by the standards of the literature.** Ours: 42,160 texts from 6 corpora
   (`results/distill3_manifest.json`). Turc et al. needed millions: "PD can match the teacher model with 10x smaller model and
   1.5x less transfer data" where the full transfer set was 8M Amazon reviews, and "distillation barely recovers teacher
   accuracy ... using the entire 8m transfer set" without pre-training (https://ar5iv.labs.arxiv.org/html/1908.08962).
   The teacher itself was trained on 1.8M public + 98k synthetic records in v0.1 plus 177k real + 78k targeted synthetic in
   v0.3 (https://huggingface.co/iapp/OpenThai-SystemOne, changelog; `openthai-systemone/docs/MODEL_CARD.md:242-270`).

4. **Domain of the transfer set matters more than its size.** Turc et al.: when transfer-set/target similarity "drops to 0.43,
   distillation on D_T is 1.8% worse than basic training", while in-domain transfer data is "consistently best"
   (https://ar5iv.labs.arxiv.org/html/1908.08962). Our wisesight texts are in the transfer set (train split, texts only,
   `thai/distill_from_ots.py:49-51`), which is why wisesight reached the teacher; sib200 (7-way news topic, Flores-derived
   sentences) has no matching source: prachathai is long news bodies, not single sentences.

5. **The encoder had no Thai-specific pre-training, but it is not Thai-starved.** mmBERT-base saw 18.2B Thai tokens (0.92%)
   in pre-training and 8.1B (1.34%) in mid-training (https://ar5iv.labs.arxiv.org/html/2509.06888, Table 9); 307M params,
   110M non-embedding (https://huggingface.co/jhu-clsp/mmBERT-base). WangchanBERTa was pre-trained on 78.5 GB of Thai
   (https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) and beat XLM-R and mBERT on wisesight/wongnai/
   prachathai-style tasks (https://arxiv.org/abs/2101.09635), but it is 0.1B params with a 416-token limit, i.e. it cannot hold
   our 768-token option budget plus state. The teacher's 5B-token Thai CPT (model card) is the one ingredient we have not
   replicated; Turc et al. show pre-training and distillation "compound" even on the same unlabelled data.

6. **Architecture limits that are not about generalisation.** (i) Each option is cut to 48 tokens and the whole option block to
   `head_max_len` (`laya/common.py:65-74`), so descriptions get truncated before the state does. (ii) The model already has a
   second head, `act_head`, fed by top-1, margin, entropy and option count (`laya/common.py:101,131-136`) with an "escalate"
   action costed at 0.5 (`out/laya-th-run3/rl_agent_config.json`), surfaced as `act_probability` (`laya/agent.py:335`).
   That is an abstain/escalate signal we never trained or used. (iii) The teacher trains abstain by dropping the correct
   option from 3% of choice questions (`openthai-systemone/scripts/04_decision_train.py:74-84`); laya has no abstain slot,
   but the same trick works as an extra option ("none of the above") because options are just marker tokens.

## (b) Ranked next experiments (one A2, 16 GB; teacher labelling ~5 rec/s at 8 workers, training ~0.54 s/step)

| # | experiment | why it should move held-out | cost | measure |
|---|---|---|---|---|
| 1 | **Drop the human-label mix, keep the 768 budget: run 2 recipe on the run 3 teacher set** (`distill3_items.pt` alone, 2 epochs) | Isolates whether the sib200 drop came from the human labels (diagnosis 2). Run 3 without them is the cleanest "bigger transfer set" test. | ~4.5 h train, 0 h labelling (set exists) | sib200, wisesight, massive_th vs run 2 and run 3 |
| 2 | **Broaden the transfer texts: 200-300k Thai texts from a general web corpus (Mangosteen / WangchanLION-Web) plus short-sentence sources, same question bank** | Turc: transfer-set size and domain drive distillation; Hinton: the transfer set "can be much larger than the original training set and can also contain examples that are not in the original training set". Our set is 42k texts from 6 corpora. Mangosteen is 47.4B tokens, 30.1M documents, permissive licences (https://arxiv.org/html/2507.14664v1; https://huggingface.co/datasets/aisingapore/WangchanLION-Web). Sample sentence-length pieces to match sib200-style inputs. | labelling 250k records / 5 rec/s = ~14 h (can run on GPU 0 while GPU 1 trains); train 2 epochs on ~700k items = ~25 h at 0.54 s/step. Split into 2-3 nights. | student-vs-teacher agreement on a fresh held-out teacher slice; sib200; the 5 tickets; a new held-out set from a corpus not in training (e.g. thai_sum headlines, LST20) |
| 3 | **Student noise: option-order re-shuffle per epoch and dropout on the student, keep the teacher clean** | Noisy Student: inject "dropout, stochastic depth, and data augmentation ... to the student so that the student generalizes better than the teacher"; gains were largest out-of-distribution (ImageNet-A 61.0 -> 83.7) (https://arxiv.org/abs/1911.04252). Upstream laya recommends "more aggressive option-order shuffling during training" for order stability (`BENCHMARKS.md`, "At 20 options"). Today items are tokenised once with one order (`thai/distill_from_ots.py:234`, `train_single.py:132`). | re-tokenise per epoch: +10-15 min/epoch; no extra labelling | order-flip rate (upstream's option-order robustness metric), sib200, ECE |
| 4 | **Thai MLM continued pre-training of mmBERT-base on the same unlabelled texts before distillation ("pre-trained distillation")** | Turc et al.: PD "consistently best"; pre-training + distillation "compound ... even when sequentially applied on the same data"; PD matched the teacher with 1.5x less transfer data than plain distillation. The teacher's own gap to us includes 5B tokens of Thai CPT. | 100-300M tokens of MLM at ~2k tok/s on an A2 in bf16 = 14-40 h; then rerun 1 (4.5 h). Biggest single bet. | MLM loss on held-out Thai; then the full eval; compare against experiment 2 at equal wall-clock |
| 5 | **Text augmentation of the transfer set (TinyBERT-style word replacement, N_a=20, p_t=0.4)** | TinyBERT's ablation: removing data augmentation cost more than removing general distillation (MNLI-m 80.5 without DA vs 82.5 without GD; "task-specific procedures (TD and DA) are more helpful than the pre-training procedure (GD)") (https://ar5iv.labs.arxiv.org/html/1909.10351). For Thai use mmBERT's own MLM head for replacements. Cheaper than experiment 2 because it reuses texts we have. | 20x augmentation of 42k texts is too much for the teacher (5 rec/s); use 3-5x = 130-210k records, ~7-12 h labelling, ~12-20 h training | same as 2 |
| 6 | **Train the `act_head` as an abstain/escalate signal and gate the cascade on it** | The head exists and reads (top-1, margin, entropy, k) (`laya/common.py:131-136`); the teacher's abstain probability is a free target for it (drop from the soft targets today, `thai/README.md` "Run 2"). Also add "none of the above" as an option on 3% of choice questions, like upstream OpenThai (`scripts/04_decision_train.py:78`). Does not raise accuracy; it makes the cascade cheaper on out-of-domain inputs where max-prob is unreliable (upstream: Khmer at 0.000 accuracy with 0.952 confidence, `research/README.md`). | code only, retrain = rerun 1 | cascade curve: accuracy vs teacher-call fraction on a held-out set; abstain AUROC vs teacher abstain |
| 7 | **Distil intermediate representations, not only outputs (MiniLM-style last-layer attention/value relations)** | MiniLM keeps ">99% accuracy ... using 50% of the Transformer parameters" (https://arxiv.org/abs/2002.10957). But teacher (decoder, Qwen3.5) and student (encoder, mmBERT) have different tokenisers and attention geometry; only pooled-representation distillation is feasible. Low expected value for Thai generalisation; listed for completeness. | new code, uncertain | agreement with teacher |

Order of attack: 1 (cheap, resolves the run 3 question) -> 3 (cheap, orthogonal) -> 2 or 4 (the real bets; 2 first because its
labelling runs on GPU 0 unattended). 5 is the fallback if the web corpus is hard to get. 6 is a cascade improvement, not a student one.

## (c) What will not help, with evidence

- **More epochs or more teacher labels from the same 6 corpora.** Run 2 -> run 3 multiplied teacher data 2.5x on the same
  corpora and held-out did not move (`results/distill.json`, `results/run3.json`); run 1 showed epochs 2-3 add nothing
  outside the training distributions (`thai/README.md`, "Run 1"). Turc: domain of the transfer set, not just size.
- **Chasing wisesight.** The student is already at or above the teacher there (0.587 / 0.557 vs 0.547); the teacher's own authors
  moved it 38.7 -> 51.5 only with a 22k synthetic Thai sentiment set (`MODEL_CARD.md:243,255`) and it has been flat since.
  A better student on wisesight needs a better teacher or human sentiment labels, which then re-introduces diagnosis 2.
- **Switching to WangchanBERTa/PhayaThaiBERT for "more Thai".** 416-token sequence limit and 0.1B params (WangchanBERTa card),
  PhayaThaiBERT 0.3B (https://huggingface.co/clicknext/phayathaibert) but same RoBERTa/CamemBERT lineage with 512 context;
  neither fits `head_max_len` 768 + state in one sequence, so the massive_th gain of run 3 would be lost. mmBERT already has
  18B Thai tokens and 8,192 context.
- **Raising `head_max_len` further.** 768 already lifted massive_th to 0.857 (teacher 0.880). Options are also capped at 48
  tokens each (`laya/common.py:65`); the budget is no longer the binding constraint on any set we measure.
- **Temperature tricks.** Run 3's fitted temperatures are 1.04 / 1.15 / 1.12 (`rl_agent_config.json`); soft targets already
  calibrate the student (ECE 0.045). Hinton's T=20 applies to hard-label teachers; our targets are the teacher's probabilities.
- **A larger student.** laya's English ModernBERT-large checkpoint (421M) collapses outside English (`BENCHMARKS.md`); no
  larger multilingual laya exists, and the 39 ms budget is the point of the student.

## (d) Open questions

- Is sib200's drop in run 3 noise? n = 204, so 0.745 -> 0.701 is 9 items. Experiment 1 answers it; also bootstrap the CI.
- Does the teacher's abstain mass carry signal for Thai out-of-domain inputs? Not measured; needed before experiment 6.
- Mangosteen/WangchanLION-Web: confirm the Hugging Face dataset id, per-source licences and whether short social/chat text is
  present (the paper lists CC, Wikipedia, YouTube subtitles, government/legal; https://arxiv.org/html/2507.14664v1).
- A/B budget: experiment 4 (Thai MLM CPT) vs experiment 2 (bigger transfer set) at equal A2-hours is the decision that matters;
  Turc's result says both, in that order, but their students were BERT-small on English reviews.
- We have no truly unseen Thai eval set that is not also a public benchmark the teacher may have seen in CPT text. Build one
  from ~300 real tickets/chats before spending 40 GPU-hours.

## Source list

- laya upstream: https://github.com/NandhaKishorM/laya (`README.md` "Honest limits", "Fine-Tuning"; `BENCHMARKS.md`;
  `research/README.md`; `laya/common.py`, `laya/agent.py`); https://huggingface.co/convaiinnovations/laya-multilingual
- OpenThai-SystemOne: https://huggingface.co/iapp/OpenThai-SystemOne; local fork `openthai-systemone/docs/MODEL_CARD.md`,
  `scripts/03_decision_data.py`, `03b_synth_generate.py`, `03d_thai_sentiment_synth.py`, `03e_targeted_synth.py`, `04_decision_train.py`
- mmBERT: https://huggingface.co/jhu-clsp/mmBERT-base; https://arxiv.org/abs/2509.06888 (Table 9 via ar5iv)
- WangchanBERTa: https://arxiv.org/abs/2101.09635; https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased;
  PhayaThaiBERT https://huggingface.co/clicknext/phayathaibert
- Distillation: Hinton et al. https://arxiv.org/abs/1503.02531; Turc et al. https://arxiv.org/abs/1908.08962;
  TinyBERT https://arxiv.org/abs/1909.10351; MiniLM https://arxiv.org/abs/2002.10957; Noisy Student https://arxiv.org/abs/1911.04252
- Thai corpora: Mangosteen https://arxiv.org/abs/2507.14664, https://github.com/vistec-AI/Mangosteen,
  https://huggingface.co/datasets/aisingapore/WangchanLION-Web; CC-100 https://huggingface.co/datasets/statmt/cc100 (unlabelled, used by XLM-R);
  OSCAR https://huggingface.co/datasets/oscar-corpus/OSCAR-2301 (research-only licence, avoid for a published model)

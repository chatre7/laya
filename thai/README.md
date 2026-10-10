# laya for Thai (fork experiment)

This branch (`thai`) of the [laya](https://github.com/NandhaKishorM/laya) fork asks whether laya's
encoder-based decision model (same choice/score/noul contract as OpenThai-SystemOne and Jev, ~40 ms
per record on an NVIDIA A2) can be made useful for Thai. Everything lives under `thai/`; the `laya/`
package is untouched so far, so upstream can still be merged.

## What is here

| file | purpose |
|---|---|
| `Dockerfile`, `run.sh` | training/eval image (the OpenThai-SystemOne serving image + `datasets`), one-line commands for the dev GPU box |
| `prep_thai.py` | human-labelled Thai data via the upstream OpenThai converters (`ots/` is copied in at run time from the OpenThai fork) |
| `distill_from_ots.py` | **distillation**: any Thai text + a bank of typed questions, labelled by the OpenThai-SystemOne server; teacher probabilities become soft targets |
| `train_single.py` | laya's fine-tune recipe (REINFORCE with a proper-scoring-rule reward + soft cross-entropy) on one GPU, bf16, gradient checkpointing, temperature calibration |
| `eval_thai.py` | accuracy on the human-labelled eval set, 5 support tickets, and (with `--teacher`) student-vs-teacher agreement |
| `eval_ots.py` | the same metrics for the OpenThai server, so both models are scored on identical records |
| `run3.sh` | run 3 end to end (bigger distillation set, option budget 768, human labels mixed in, train, eval), unattended |
| `cascade.py`, `cascade3.sh` | student -> teacher cascade sweep on the human-labelled decisions (accuracy vs teacher-call fraction per confidence threshold) |
| `cascade_server.py`, `Dockerfile.cascade`, `docker-compose.cascade.yml`, `smoke_cascade.py`, `bench_cascade.py`, `probe_other.py` | **the cascade as a service**: same `/v1/systemone` contract as the teacher, student on GPU 1 at `:8011`, teacher at `:8010`; smoke test with a 60-option question and 8 concurrent callers |
| `rewrite_colloquial.py`, `check_rewrites.py`, `run_rewrite.sh` | run 4 data: rewrite the Thai Bitext customer-support set into spoken/chat Thai with a local LLM (vLLM), then let the teacher check that each rewrite still carries its intent |
| `cs/` | run 6: `cs_questions.py` (business + per-business intent lists for telecom / banking / insurance / e-commerce, shared questions), `fetch_bitext_cs.py`, `translate_colloquial.py` (EN -> Thai customer message), `check_cs_rewrites.py` (teacher gate), `prep_cs_data.sh`, `label_cs.py`, `run6.sh`, `probe_cs.py` |
| `label_cc.py`, `run4.sh`, `cascade4.sh`, `run5.sh` | runs 4-5: the call-center question set labelled by two teacher instances, items, grouped eval split, train from run 3, eval |
| `research-generalisation.md` | research note: why the held-out sets are flat and what could move them (ranked, with sources) |
| `results/` | json summaries and the training log of every run |

```bash
bash thai/run.sh build
bash thai/run.sh prep 1500                                   # human-labelled set (needs thai/ots)
bash thai/run.sh distill --per-source 3000                   # teacher-labelled set, background
bash thai/run.sh train-distill --epochs 2                    # background
bash thai/run.sh eval-distill /work/thai/out/laya-th-distill distill
nohup bash thai/run3.sh > thai/out/run3.log 2>&1 &            # run 3, ~6 h on one A2
```

## Run 1: supervised fine-tune on 5 public Thai datasets (2026-09-22)

10,674 sequences (wongnai, prachathai, xnli_th, massive_th, thai_toxicity; 1,500 records per split),
3 epochs, 55 min on one A2. Eval on 1,704 records; wisesight and sib200_th are held-out for both models.

| source : type | n | laya-multilingual base | + Thai fine-tune | OpenThai-SystemOne |
|---|---|---|---|---|
| massive_th : choice | 300 | 0.347 | 0.610 | 0.880 |
| prachathai : choice | 163 | 0.362 | 0.804 | 0.969 |
| prachathai : noul | 588 | 0.558 | 0.861 | 0.940 |
| xnli_th : choice | 300 | 0.723 | 0.753 | 0.823 |
| xnli_th : noul | 300 | 0.843 | 0.843 | 0.867 |
| wongnai : score (exact / MAE) | 300 | 0.257 / 0.97 | 0.573 / 0.49 | 0.642 / 0.41 |
| **wisesight : choice (held-out)** | 300 | 0.290 | 0.273 | 0.547 |
| **sib200_th : choice (held-out)** | 204 | 0.755 | 0.740 | 0.784 |
| tickets: department / refund / frustration MAE | 5 | 3/5, 4/5, 0.86 | 3/5, 4/5, 0.82 | 5/5, 5/5, 0.39 |
| latency per record (A2) | | 44 ms | 39 ms | ~100 ms |

Large in-domain gains, no generalisation: the held-out sets and the tickets do not move, and the fitted
temperatures (3.7 / 1.4 / 4.7) show the fine-tuned logits are badly over-confident. The model learned five
training distributions, not Thai decisions. Details and the reading in the System One repo's `laya-ft/README.md`.

## Run 2: distillation from OpenThai-SystemOne

> Teacher version: OpenThai-SystemOne **v0.3** (HF `f3709948`) for runs 2-4 and for every teacher number below; the serving container
> picked up v0.3 at its 2026-09-22 rebuild (verified 2026-09-24, byte-identical outputs). v0.2 on the same 2,455 decisions scores 0.816 vs
> v0.3 0.814 overall, with massive_th +1.7 / sib200 +4.4 for v0.3 and wongnai -2.5 / xnli -0.4, -1.3 (`results/ots_v02.json`).

Instead of human labels, `distill_from_ots.py` sends Thai texts from six public corpora with 2–4
questions each from a bank of ~20 question types (sentiment, intent, department, urgency, frustration,
topic, rating, NLI, several yes/no checks; paraphrased instructions, option subsets of varying size) to the
OpenThai-SystemOne server and keeps its probabilities as soft targets. The teacher answers at
~100 ms per record, so the set costs GPU time, not annotation. The student can at best match the
teacher, but at ~40 ms and 322M parameters. It has no abstain slot, so the teacher's abstain mass is
dropped and the remaining probabilities renormalised.

Data: 16,964 Thai texts (6 corpora, 3,000 each) x 2-4 questions -> 47,379 sequences, labelled in 29 min at 8
concurrent requests. Training: 2 epochs, 98 min on one A2, fitted temperatures 1.03 / 1.22 / 0.95 (run 1: 3.7 / 1.4 / 4.7,
i.e. soft targets fix the over-confidence by themselves).

| source : type | n | laya base | run 1 (supervised) | **run 2 (distilled)** | OpenThai teacher |
|---|---|---|---|---|---|
| massive_th : choice | 300 | 0.347 | 0.610 | 0.540 | 0.880 |
| prachathai : choice | 163 | 0.362 | 0.804 | 0.767 | 0.969 |
| prachathai : noul | 588 | 0.558 | 0.861 | 0.806 | 0.940 |
| xnli_th : choice | 300 | 0.723 | 0.753 | 0.770 | 0.823 |
| xnli_th : noul | 300 | 0.843 | 0.843 | 0.847 | 0.867 |
| wongnai : score (exact / MAE) | 300 | 0.257 / 0.97 | 0.573 / 0.49 | 0.633 / 0.44 | 0.642 / 0.41 |
| **wisesight : choice (held-out)** | 300 | 0.290 | 0.273 | **0.587** | 0.547 |
| **sib200_th : choice (held-out)** | 204 | 0.755 | 0.740 | 0.745 | 0.784 |
| tickets: department / refund / frustration MAE | 5 | 3/5, 4/5, 0.86 | 3/5, 4/5, 0.82 | 4/5, 4/5, 0.31 | 5/5, 5/5, 0.39 |
| overall accuracy / Brier / ECE (2,455 decisions) | | | | 0.719 / 0.400 / 0.054 | |
| latency per record (A2) | | 44 ms | 39 ms | 38 ms | ~100 ms |

Student vs teacher on 848 held-out teacher-labelled records: argmax agreement choice 0.759,
noul 0.933, score 0.746; mean total-variation distance
0.231 / 0.080 / 0.189.

**Same data on Kaggle 2xT4** (`kaggle_notebook/`, a copy of batprem's public notebook pointed at the dataset
`chatre7/laya-thai-distill`, fp16 + DDP, effective batch 64, its own 5% split as val): 1 h 45 min wall (~1 s/step for
16 sequences, i.e. the same throughput as one A2), fitted temperatures 1.17 / 1.19 / 1.17. Its own held-out
(805 teacher-labelled cases): base 0.563 -> fine-tuned 0.811 accuracy, Brier 0.385 -> 0.110, ECE 0.194 -> 0.056.
Evaluated with our script on the same records as the A2 run: overall 0.697 / Brier 0.418 / ECE 0.041,
wisesight 0.553, sib200 0.735, massive 0.500, tickets 4/5, 5/5, MAE 0.31;
teacher agreement 0.752 / 0.916 / 0.751. Slightly below the A2 run everywhere
(half the optimizer updates at twice the batch, 5% less data), so the A2 checkpoint is the one published as
[Chatre7/laya-thai-distill](https://huggingface.co/Chatre7/laya-thai-distill) (private). `results/kaggle.json`, `results/kaggle_v2.log`.

**Reading.** Distillation did what supervised fine-tuning could not: the held-out sets moved (wisesight 0.27 -> 0.59,
above the teacher's 0.55 on that set; sib200 within 4 points of the teacher), the tickets now behave (department 4/5,
frustration MAE 0.31 vs the teacher's 0.39, score no longer stuck at ~1.2) and calibration is good (ECE 0.054) without
any post-hoc temperature. Where it stays far behind the teacher is massive_th (0.54 vs 0.88): those
questions carry up to 60 intent options, which laya's 256-token option budget squeezes to ~4 tokens each. The
in-domain sources of run 1 (prachathai, wongnai) are a little lower than run 1 because run 2 never saw their labels,
only the teacher's opinion of similar texts.

## Run 3: bigger distillation set + human labels, option budget 768 (2026-09-23)

`run3.sh`, unattended on GPU 1 (6 h 10 min wall). Changes from run 2: 8,000 texts per source instead of 3,000
(42,160 texts, 117,465 teacher-labelled sequences, 77 min at 8 concurrent requests, 5.1 rec/s), a bank of
wide-intent questions aimed at massive_th, the option budget raised from 256 to 768 tokens (`head_max_len`),
and the 10,674 human-labelled sequences of run 1 re-tokenised at 768 and mixed in (128,139 items in all).
Order-invariant teacher answers were tried and dropped: labelling fell to 2 rec/s for no visible gain. 2 epochs,
8,008 updates, 0.54 s/step, 286 min; fitted temperatures 1.04 / 1.15 / 1.12.

| source : type | n | run 1 (supervised) | run 2 (distilled) | **run 3 (distilled + human)** | OpenThai teacher |
|---|---|---|---|---|---|
| massive_th : choice | 300 | 0.610 | 0.540 | **0.857** | 0.880 |
| prachathai : choice | 163 | 0.804 | 0.767 | **0.865** | 0.969 |
| prachathai : noul | 588 | 0.861 | 0.806 | **0.900** | 0.940 |
| xnli_th : choice | 300 | 0.753 | 0.770 | 0.743 | 0.823 |
| xnli_th : noul | 300 | 0.843 | 0.847 | 0.830 | 0.867 |
| wongnai : score (exact / MAE) | 300 | 0.573 / 0.49 | 0.633 / 0.44 | 0.623 / 0.43 | 0.642 / 0.41 |
| **wisesight : choice (held-out)** | 300 | 0.273 | 0.587 | 0.557 | 0.547 |
| **sib200_th : choice (held-out)** | 204 | 0.740 | 0.745 | 0.701 | 0.784 |
| tickets: department / refund / frustration MAE | 5 | 3/5, 4/5, 0.82 | 4/5, 4/5, 0.31 | 4/5, 5/5, 0.25 | 5/5, 5/5, 0.39 |
| overall accuracy / Brier / ECE (2,455 decisions) | | | 0.719 / 0.400 / 0.054 | **0.772 / 0.314 / 0.045** | |
| latency per record (A2) | | 39 ms | 38 ms | 38 ms | ~100 ms |

Student vs teacher on 2,108 held-out teacher-labelled records: argmax agreement choice 0.765, noul 0.927,
score 0.789; mean total-variation distance 0.218 / 0.080 / 0.171 (run 2: 0.759 / 0.933 / 0.746). `results/run3.json`,
`results/run3.log`, `results/distill3_manifest.json`. Published as [Chatre7/laya-thai-distill-run3](https://huggingface.co/Chatre7/laya-thai-distill-run3) (private); run 2 stays at `Chatre7/laya-thai-distill`.

**Reading.** Overall accuracy 0.719 -> 0.772 (teacher 0.814) and Brier 0.400 -> 0.314, with calibration still good.
The gain is in-domain: massive_th 0.54 -> 0.86 (the 768-token option budget plus wide-intent questions did what
they were meant to), prachathai +10 points from seeing its human labels again. The held-out sets did not improve:
wisesight 0.587 -> 0.557 (still at the teacher's 0.547), sib200 0.745 -> 0.701, so the extra 2.5x teacher data and the
human labels bought no new generalisation, and sib200 hints at mild over-fitting to the training corpora. Tickets:
the same login ticket is still routed to billing (p=0.70); everything else is right and frustration MAE is the best so far.

**Cascade with this checkpoint** (`cascade3.sh`, same 2,455 decisions, teacher 0.815 alone; `results/cascade3.json`,
`results/cascade3_opt60.json`). Left: the run 2 rule (questions with more than 10 options always go to the teacher).
Right: option gate off (`--max-options 60`), since this student has the 768-token option budget.

| confidence threshold | run 2 cascade: acc / to teacher | **run 3, max-options 10: acc / to teacher** | run 3, option gate off: acc / to teacher |
|---|---|---|---|
| 0.00 | 0.758 / 12% | 0.780 / 12% | 0.772 / 0% |
| 0.50 | 0.774 / 22% | 0.791 / 20% | 0.785 / 8% |
| 0.60 | 0.782 / 34% | **0.801 / 29%** | 0.797 / 18% |
| 0.70 | 0.793 / 46% | **0.804 / 38%** | **0.803 / 28%** |
| 0.80 | 0.795 / 58% | 0.809 / 50% | 0.809 / 40% |
| 0.90 | 0.805 / 73% | 0.813 / 62% | 0.813 / 53% |
| teacher only | 0.814 / 100% | 0.815 / 100% | 0.816 / 100% |

Better at every point: threshold 0.7 now gives 0.804 (98.7% of the teacher) with 38% teacher calls instead of
0.793 with 46%; with the option gate off, 0.803 with 28% of calls, i.e. ~3.5x the teacher-only throughput
for a 1.3-point loss. The option-count rule no longer pays: the student's massive_th (up to 60 intents) is 0.857 on its
own, and at threshold 0 the two rules differ by 0.8 points for 12% of teacher calls. Teacher batched latency under
8 concurrent callers was ~880 ms again, so the estimated per-record latency at 0.7 / gate off is ~283 ms vs ~912 ms teacher-only.

## Using the speed: student -> teacher cascade (`cascade.py`)

The student answers every decision; the teacher is called when the question has more than 10 options
(the student's option budget) or the student's confidence (max probability) is below a threshold. Measured on
the 2,455 human-labelled decisions with both models on the dev box:

| confidence threshold | cascade accuracy | sent to teacher | student acc on kept | teacher acc on sent | est. latency per record* |
|---|---|---|---|---|---|
| 0.00 | 0.758 | 12% | 0.738 | 0.902 | ~51 ms |
| 0.50 | 0.774 | 22% | 0.768 | 0.795 | ~61 ms |
| 0.60 | 0.782 | 34% | 0.798 | 0.750 | ~73 ms |
| 0.70 | 0.793 | 46% | 0.834 | 0.746 | ~85 ms |
| 0.80 | 0.795 | 58% | 0.866 | 0.744 | ~97 ms |
| 0.90 | 0.805 | 73% | 0.913 | 0.765 | ~112 ms |
| 0.95 | 0.809 | 82% | 0.949 | 0.779 | ~121 ms |
| 1.01 | 0.814 | 100% | 0.000 | 0.814 | ~139 ms |

\* 39 ms student + fraction x ~100 ms teacher (single request; the teacher's batched latency under 8 concurrent
callers was 883 ms in this run). Student alone 0.719, teacher alone 0.814.

- The option-count rule alone (threshold 0) recovers most of the gap: 0.719 -> 0.758 with 12.5% teacher calls.
- Threshold 0.7 gives 0.793 (97% of the teacher's accuracy) with 46% teacher calls, i.e. roughly 2x the
  teacher-only throughput on the same GPU.
- Above 0.8 the student keeps only its easy decisions and the cascade converges to the teacher; not worth it.
- The student is well calibrated (ECE 0.05), which is what makes the confidence gate usable at all.

## Serving the cascade (`cascade_server.py`, port 8011)

Deployed on the dev box next to the teacher (2026-09-23 with run 3, run 4 from 2026-09-24 18:40, **run 5 since 2026-09-25 09:50**): `docker compose -f thai/docker-compose.cascade.yml up -d --build`
builds `laya-cascade:run4` from the training image, mounts `thai/out/laya-th-run4` read-only, GPU 1 (~1.6 GB), `restart: unless-stopped`.
With run 4 the student keeps more decisions (it is more confident): the same ticket smoke test sends only `frustration` to the teacher,
the 60-intent question stays with the student (p 1.00), and the nonsense input "อืม" now also stays with the student (p 0.73) where run 3
sent it to the teacher and got abstain 0.98. Raise `CASCADE_THRESHOLD` if unanswerable inputs must reach the teacher's abstain.

- `POST http://172.18.72.145:8011/v1/systemone`: the same request and response as the teacher at `:8010` (state, typed
  questions, `model`, `order_invariant` / `permutations` are passed through to the teacher for the questions it answers).
  The student answers all questions in one forward pass; every question whose max probability is below
  `CASCADE_THRESHOLD` (0.7) goes to the teacher in one follow-up request with the same state. `usage.cascade` lists
  the questions that reached the teacher and why (`confidence<0.70`, `options>N`, `student_error`), plus both latencies.
  Teacher answers keep their `abstain`; student answers have none. If the teacher is down, the student's low-confidence
  answer is returned with reason `...;teacher_unavailable` instead of failing.
- `GET /healthz` (student loaded + teacher `/healthz`), `GET /stats` (teacher-question fraction, mean latencies), `/docs`.
- Routing lives in one function, `route()`: one global threshold and an optional option-count gate (`CASCADE_MAX_OPTIONS`,
  0 = off as measured). Per-type thresholds or a per-request override go there.
- **Student batching + torch.compile (2026-09-24 evening).** The student now has a dynamic batcher (one GPU thread; requests
  arriving within `STUDENT_MAX_WAIT_MS`=6 are collated into one forward, at most `STUDENT_MAX_BATCH`=32 requests /
  `STUDENT_MAX_BATCH_TOKENS`=24576 padded tokens; decode uses laya's own formula, so answers match single requests: argmax
  64/64, max probability difference 0.015 from bf16) and, with `STUDENT_COMPILE=1`, `torch.compile(dynamic=True)` with a
  75-shape warm-up at start so live requests do not pay the compile (start-up 200 s instead of 25 s; set 0 to disable).
  `bench_cascade.py`, same script before and after (`results/bench_before*.txt`, `bench_batched.txt`, `bench_compiled*.txt`, `bench_final.txt`):

  | path | concurrent | before: mean ms / req/s | batching | batching + compile |
  |---|---|---|---|---|
  | student only (60-intent question) | 1 | 56 / 18 | 72 / 14 | **52 / 19** |
  | | 8 | 345 / 22 | 230 / 34 | **189 / 41** |
  | | 32 | 1,067 / 28 | 826 / 38 | **668 / 47** |
  | ticket (3 questions, one to the teacher) | 1 | 151 / 6.6 | 169 / 5.9 | 158 / 6.3 |
  | | 8 | 315 / 25 | 255 / 30 | **242 / 31** |
  | | 32 | 1,207 / 25 | 640 / 48 | **575 / 54** |

  About 2x under load, not the 3x the teacher's batcher gave: the 60-option question is ~800 tokens, so a batch of 30 is
  24k tokens and the A2 is compute-bound (batching only removes launch overhead and idle gaps). Shorter questions batch better
  (ticket path 2.1x). The next lever is the forward itself (fp16 TensorRT export) or shorter option lists.
- Smoke test (`smoke_cascade.py`, from a LAN machine): ticket 3 questions -> department and refund kept by the student,
  frustration (score, max p 0.42) sent to the teacher, 130-450 ms; a 60-intent question answered by the student alone
  in 84 ms (max p 0.98); "อืม" sent to the teacher, which returns abstain 0.98. 32 ticket requests at 8 concurrent:
  mean 368 ms, p95 485 ms, 20.5 req/s (teacher-only under the same load: ~880 ms). The student runs one forward at a
  time behind a lock, so its share grows with concurrency; batching it is the next step if 8011 gets real traffic.

## Run 4: call-center distillation (2026-09-24)

Runs 1-3 chase public benchmarks; run 4 targets the job the student is for: call-center triage (route, urgency,
frustration, yes/no checks, intent). There is no downloadable Thai call-center corpus on Hugging Face (the AIxBlock and
Nexdata listings are sales samples, non-commercial), so the data is built:

1. **Source**: [`Porameht/customer-support-th-26.9k`](https://huggingface.co/datasets/Porameht/customer-support-th-26.9k)
   (cc-by-sa-3.0), the Bitext customer-support set localised to Thai: 26,872 utterances, 27 intents, 11 categories. It is
   written Thai ("ฉันต้องการยกเลิกคำสั่งซื้อ"), not what callers say.
2. **Colloquial rewrite** (`rewrite_colloquial.py`): Qwen3-4B on vLLM (GPU 1, 32 concurrent, 3.2 rewrites/s) writes each
   utterance 3 times in different registers (hurried chat, call transcript, angry, very polite, teen/social), with a fixed
   speaker (ผม/ครับ, ฉัน/ค่ะ, หนู/ค่ะ) and placeholders filled. Typhoon 2.5 (4B) was tried on the same 300 and was worse
   (76% vs 79% label agreement, degenerate outputs), Qwen3-4B kept. 80,616 -> 72,542 after the output filter (repeats,
   non-Thai, prompt leaks), 7 h.
3. **Teacher gate** (`check_rewrites.py`): OpenThai-SystemOne answers the 27-way intent question on every rewrite;
   agreement with the source label 81.6%, the same as on the formal sources (83%), i.e. the rewrite costs almost nothing.
   Kept p(label) >= 0.5: **57,635 utterances** (79%), every intent >= 1,300, mean 93 characters. The "angry" register
   loses most (75%) because the LLM turns requests into complaints, which the teacher correctly relabels. 3 h 20 min.
4. **Question set** (`label_cc.py`, the contract the student is trained for): `intent` (27), `category` (11),
   `department` (billing / shipping / account / sales / support), `urgency` (0-2), `frustration` (0-2), `wants_refund`,
   `wants_human`, `has_order_ref` (noul), `sentiment` (4). Human labels where they exist (intent, category; sentiment for
   the 12,000 wisesight texts added as real Thai) are one-hot targets, the rest are the teacher's probabilities. Two teacher
   instances (GPU 0 :8010, GPU 1 :8013) label 69.6k texts x 9 questions in one request each.
5. **Train** (`run4.sh`): from the run 3 checkpoint, 2 items per text (the human-labelled question + one sampled) plus the
   run 1 human items, 2 epochs. **Eval**: 5% of the texts held out by source sentence (no paraphrase of an eval sentence in
   train), human labels for intent/category/sentiment plus student-vs-teacher agreement on the rest; run 3 is scored on the
   same set as the baseline, and the old public eval set is re-run as a regression check.

Published as [Chatre7/laya-thai-callcenter](https://huggingface.co/Chatre7/laya-thai-callcenter) (private).

**Results** (2026-09-24; `results/run4_cc.json`, `run3_cc.json`, `run4.json`, `cascade4*.json`). Labelling 69,635 texts x 9
questions took 2 h 32 min with two teachers (7.7 rec/s); training 132,304 cc items + 10,674 human items, 2 epochs, 4 h 38 min from
the run 3 checkpoint, fitted temperatures 1.60 / 1.03 / 1.03 (choice over-confident after one-hot intent labels). A first attempt
was stopped after 50 min because the item builder always picked `intent` as the labelled question, so `category` was never trained;
fixed by picking the labelled question at random (`run4_attempt1.log`).

Held-out call-center set: 3,483 texts (5% of source sentences, so no paraphrase of an eval sentence is in train), 6,375 human-labelled
decisions (intent + category on the cc rows, sentiment on wisesight rows), teacher agreement on the other 6 questions.

| | run 3 (before) | **run 4** | teacher (v0.3) |
|---|---|---|---|
| cc : intent + category (human labels, 5,784) | 0.762 | **0.998** | 0.898 (all 6,375) |
| wisesight : sentiment (human labels, 591) | 0.519 | **0.734** | |
| overall accuracy / Brier / ECE | 0.740 / 0.466 / 0.230 | **0.974 / 0.039 / 0.010** | |
| agreement with the teacher: choice / noul / score | 0.713 / 0.929 / 0.492 | **0.905 / 0.987 / 0.840** | |
| mean TV distance: choice / noul / score | 0.431 / 0.140 / 0.283 | 0.133 / 0.027 / 0.106 | |
| 5 tickets: department / refund / frustration MAE | 4/5, 5/5, 0.25 | **5/5**, 5/5, 0.48 | 5/5, 5/5, 0.39 |
| latency per record (A2, 9 questions) | 61 ms | 61 ms | ~1,280 ms at 8 concurrent |

Read 0.998 with care: Bitext's utterances are template-heavy within an intent, so eval sentences still resemble train sentences.
The agreement on the unlabelled questions (urgency, frustration, wants_refund, wants_human, has_order_ref) is the better signal: the
student now reproduces the teacher's call-center judgements at 0.84-0.99 argmax agreement instead of 0.49-0.93. On the human-labelled
questions the student beats the teacher (0.998 vs 0.898) because it saw labels the teacher never had.

Public eval set (regression check, run 3 -> run 4): overall 0.772 -> **0.783**; wisesight 0.557 -> 0.720 (no longer held-out: its
train split was in the mix), sib200 (still held-out) 0.701 -> 0.725, prachathai choice 0.865 -> 0.883, but massive_th 0.857 -> 0.820,
xnli 0.743 / 0.830 -> 0.713 / 0.833, wongnai 0.623 -> 0.613, and calibration worse (ECE 0.045 -> 0.132). The login ticket is finally
routed to `technical` (p 0.64); frustration on the tickets is worse than run 3.

Cascade with the run 4 student (`cascade4.sh`, option gate off): on the call-center set the student alone (0.974) beats the teacher
(0.898), so the gate should stay near 0 there (threshold 0.7 sends 2.5% and loses 0.4 points). On the public set threshold 0.7 gives
0.794 with 16% teacher calls (run 3: 0.803 with 28%): the student is more confident and the teacher rescues less of what it sends
(teacher accuracy on sent items 0.60). One threshold no longer fits both: per-question-set or per-source thresholds are the next step.

## Run 5: calibration, more score items, an `other` intent (2026-09-25)

Same labelled records as run 4, no new teacher labelling (`run5.sh`, `label_cc.py --from-records --prefix cc5`), three changes aimed
at run 4's weak spots: **3 items per text** instead of 2 (score items 16.6k -> 33k, noul 25k -> 50k; 198,456 items), **label
smoothing 0.1** on the human one-hot targets, and an **`other` option on the intent question** with the 12,000 wisesight texts
labelled `other` (out-of-scope examples: laya has no abstain, so the student learns "none of these" as an option). Trained from
the run 3 checkpoint, 2 epochs, 6 h 25 min; fitted temperatures 1.05 / 0.98 / 1.01 (run 4: 1.60 / 1.03 / 1.03).

Held-out set re-cut with the new intent question (`cc5_eval_human.jsonl`: 3,483 texts; the wisesight rows now carry two human labels,
sentiment and intent = `other`), all three checkpoints scored on it:

| | run 3 | run 4 | **run 5** |
|---|---|---|---|
| cc : intent + category (human labels, 5,784) | 0.745 | 0.998 | **0.998** |
| wisesight : sentiment + intent=`other` (human labels, 1,182) | 0.285 | 0.525 | **0.873** |
| overall accuracy / Brier / ECE (6,966) | 0.667 / 0.523 / 0.198 | 0.917 / 0.121 / 0.043 | **0.977 / 0.045** / 0.092 |
| agreement with the teacher: choice / noul / score | 0.698 / 0.929 / 0.492 | 0.897 / 0.987 / 0.840 | **0.933 / 0.989 / 0.861** |
| 5 tickets: department / refund / frustration MAE | 4/5, 5/5, 0.25 | 5/5, 5/5, 0.48 | 5/5, 5/5, **0.40** |

Run 4's 0.525 on the wisesight rows is mostly the `other` question it never saw (its sentiment alone was 0.734). Run 5 answers `other`
on 87% of out-of-scope texts, and the teacher (which has abstain instead of an `other` option) scores 0.822 on this set, so the
student alone is the better call-center router (`cascade5_cc.json`: threshold 0.7 sends 1.4% and gains nothing).

Public set (`run5.json`): overall 0.782 (run 4 0.783, run 3 0.771); massive_th 0.827, xnli 0.733 / 0.833 recover a little from run 4,
wongnai 0.590 slips, sib200 0.711, wisesight 0.723. Calibration on the public set is not better (ECE 0.151 vs run 4 0.132): the
smoothing applied to the call-center targets, while the public questions come from the unsmoothed run 1 human items. The public-set
cascade no longer helps (threshold 0.7: 0.777 with 11% teacher calls vs student alone 0.782), i.e. for questions outside the
call-center set send to the teacher by question type, not by the student's confidence.

Published as [Chatre7/laya-thai-callcenter](https://huggingface.co/Chatre7/laya-thai-callcenter) revision `9864d7fd` (run 4 is the
previous commit `2a036745` of the same repo). Serving at `:8011` since 2026-09-25 09:50; `probe_other.py` checks the `other` option on the endpoint.

Reading: run 5 is the checkpoint to serve for call-center triage (same in-domain accuracy as run 4, out-of-scope detection, better
score agreement, temperatures near 1) and equal to run 4 elsewhere. Still unmeasured on real tickets.

## Real-ticket eval set: the review workflow (`real_eval/`)

Every number above comes from synthetic text. The tools to replace it with real tickets (run from any machine on the LAN, no GPU):

1. `python thai/real_eval/draft_labels.py --inp tickets.txt --out tickets_review.xlsx`: one text per line (or a .csv with a `text`
   column / .jsonl with `text`). The teacher (`:8010`) drafts all 9 answers, the student (`:8011`) answers alongside; the sheet
   has a dropdown per question, prefilled with the teacher's draft, cells where teacher and student disagree highlighted and those
   rows sorted first, a `questions` tab with the option descriptions, and hidden helper columns (student answer, both confidences).
   A reviewer corrects the drafts in Excel; empty cells are treated as "not labelled".
2. `python thai/real_eval/score_real.py --sheet tickets_review.xlsx --out thai/real_eval/tickets_eval.jsonl --model student=http://172.18.72.145:8011 --model teacher=http://172.18.72.145:8010`:
   per-question accuracy (exact level + MAE for scores) for each endpoint against the reviewed labels, and the eval set in
   `eval_thai.py` format so future checkpoints can be scored offline on the GPU box.

`demo_texts.txt` (20 made-up tickets) -> `demo_review.xlsx` -> `demo_eval.jsonl` show the round trip; scored against the
teacher's own unreviewed drafts the student agrees on 84% (intent 58%, department 89%, noul 95-100%), and the disagreements
are exactly what a reviewer should decide: "วันนี้อากาศดีจัง" teacher says `newsletter_subscription`, student says `other`;
"แอปล็อกอินไม่ได้" teacher `recover_password`, student `other`; "โกรธมาก โทรสามรอบไม่มีใครรับ" teacher `complaint`, student
`get_refund`. The question set is in `cc_questions.py` (shared with `label_cc.py`); extend it there when the call center's real
intents differ from Bitext's 27, then re-run `label_cc.py` and train.

## First real data, and the customer-service domain (telecom / banking / insurance) (2026-09-25)

Three real-style files arrived (`System One/data/`): 200 e-commerce support messages, 154 refund messages, 180 debt-collection
replies. Drafted with `real_eval/draft_labels.py` (sheets in the same folder, awaiting review). What they showed:

- The shared questions transfer: department 0.88, frustration 0.92-0.96, sentiment 0.84-0.95, wants_refund / has_order_ref 0.95-1.00
  teacher-student agreement on texts neither model was trained on.
- **Intent does not**: the Bitext 27 cover about half of real e-commerce messages (student `other` 104/200, teacher `complaint`
  61/200, agreement 0.39), and on debt collection (a different domain, its own question set `cc_questions.DEBT_QUESTIONS`) the
  student says `other` 116/180 because its `other` means "not customer support". Intent lists must be per domain, and the
  student must see each domain in training. `real_eval/intents_proposal.md` has an e-commerce list drafted from the messages.
- The call center is actually **telecom, banking and insurance** (debt collection kept as a separate domain for later), so the
  customer-service question set is being rebuilt around those: `cs/cs_questions.py` = a `business` question (telecom / banking /
  insurance / other) + one intent list per business from the Bitext telco (26), retail-banking (26) and insurance (39) sets with
  Thai descriptions, + the shared questions (department now has `technical` and `claims`, `has_reference` covers account / policy /
  claim numbers). `cs/prep_cs_data.sh` builds the training text the same way as run 4: 300 English utterances per intent (27,300),
  Qwen3-4B writes each as a Thai customer message in 2 registers with a fixed speaker (`translate_colloquial.py`), the teacher
  keeps the ones whose intent it still recognises (`check_cs_rewrites.py`). Launched 2026-09-25 11:30, ~7 h. Next: label the
  kept texts with the full set (human intent + business one-hot, teacher for the rest), train run 6 from run 3, score on the
  reviewed real sheets.

## Run 6: telecom / banking / insurance / e-commerce (2026-09-26)

Data (`cs/prep_cs_data.sh`, 2026-09-25 11:30-20:59): 27,300 English utterances (300 per intent from the Bitext telco, retail-banking
and insurance sets) -> 54,586 Thai customer messages by Qwen3-4B in 2 registers -> teacher gate kept **40,608** (74%; intent agreement
telecom 0.78, banking / insurance similar; the "angry" register loses most again). Plus the 57,635 e-commerce rewrites of run 4 with
Bitext's 27 intents mapped onto the 21-intent e-commerce list (`cs_questions.BITEXT_TO_ECOM`) and 12,000 wisesight posts as
out-of-scope. `cs/label_cs.py`: 110,243 texts, business + intent one-hot (smoothing 0.1), teacher for the shared questions, two
teachers, 4 h 24 min; 3 items per text = **314,133 items** + the run 1 human items. Train from run 3, 2 epochs, 20,300 updates,
**10 h 25 min**; temperatures 1.04 / 1.09 / 1.02. Chained automatically after the data prep (`cs/run6.sh`).

Held-out set (5% by source sentence, 5,532 texts, 11,665 human-labelled decisions), all three checkpoints on it:

| | run 3 | run 5 | **run 6** |
|---|---|---|---|
| banking intent (1,230) | 0.706 | 0.815 | **0.981** |
| insurance intent (1,714) | 0.623 | 0.738 | **0.962** |
| telecom intent (1,188) | 0.588 | 0.760 | **0.944** |
| e-commerce intent, 21-intent list (5,730) | 0.258 | 0.533 | **0.996** |
| out-of-scope: sentiment + intent=`other` (1,803) | 0.437 | 0.916 | 0.895 |
| overall accuracy / Brier / ECE | 0.420 / 0.769 / 0.104 | 0.675 / 0.543 / 0.182 | **0.969 / 0.055 / 0.082** |
| agreement with the teacher: choice / noul / score | 0.515 / 0.883 / 0.449 | 0.721 / 0.961 / 0.800 | **0.928 / 0.992 / 0.873** |
| 5 tickets: department / refund / frustration MAE | 4/5, 5/5, 0.25 | 5/5, 5/5, 0.40 | 5/5, 5/5, 0.49 |

Public set: 0.785 (run 5 0.782): prachathai 0.902 / 0.895 and xnli 0.737 / 0.843 up, wisesight 0.673 (from 0.723) and sib200 0.691
(from 0.711) down, ECE 0.153. `results/run6*.json`, `run*_cs6.json`, `cs6_manifest.json`, `cs_prep.log`.

**Hand-written probe** (`cs/probe_cs.py`, 13 natural Thai messages outside the templates): intent right on 8, including
cancel_transfer, apply_for_mortgage, downgrade_coverage, damaged_or_wrong_item, track_order, block_card; `other` on the two nonsense
inputs; misses on roaming, file_claim, track_claim (-> `other`). But **`business` answered `other` on 6 clearly in-domain messages**:
its `other` examples are real posts (wisesight) while every in-domain example is LLM-written, so it partly learned register, not
business. Do not ask `business` in production; the caller knows the line of business, ask `intent` with that list. Fix for run 7:
in-domain `other` texts of the same register (e.g. LLM-written off-topic chatter) and the reviewed real sheets as human items.

Served at `:8011` since 2026-09-26 15:15 (batching + compile, 58.7 req/s on the ticket smoke test). Published as
[Chatre7/laya-thai-callcenter](https://huggingface.co/Chatre7/laya-thai-callcenter) revision `6a4dc84a` (run 5 = `9864d7fd`, run 4 = `2a036745`).

## Run 7: in-register out-of-scope texts (2026-09-27)

`cs/gen_other.sh`: Qwen3-4B writes 5,020 off-topic messages in the same customer registers (small talk, other businesses, nonsense,
9 kinds), labelled `business=other, intent=other`, added to the run 6 records (`label_cs.py --from-records cs6`, only the new texts
go to the teacher, 27 min). 328,200 items, trained from run 3 like run 6, finished 07:52. Held-out (re-cut, 5,863 texts):

| | run 5 | run 6 | **run 7** |
|---|---|---|---|
| in-register out-of-scope (530) | 0.800 | 0.619 | **0.983** |
| banking / insurance / telecom intent | 0.82 / 0.74 / 0.80 | 0.99 / 0.97 / 0.95 | 0.98 / 0.95 / 0.96 |
| e-commerce intent (5,970) | 0.572 | 0.998 | 0.996 |
| overall accuracy / Brier / ECE | 0.702 / 0.504 / 0.152 | 0.961 / 0.071 / 0.075 | **0.970 / 0.053 / 0.085** |
| public set | 0.782 | 0.785 | 0.782 |

So the register confound is fixed on synthetic text. Served at `:8011` since 2026-09-27 (`laya-cascade:run7`). `results/run7*.json`,
`run*_cs7.json`, `cs7_manifest.json`, `gen_other.log`.

## The real-text check: Pantip (2026-09-26/27)

Real Thai in-domain text with trustworthy labels did not exist, so I built some: `cs/filter_pantip.py` keeps Pantip topics about
mobile / internet providers (2,889), banks (1,681) and insurers (365) from the public `pantip` dump; 120 per business were labelled by
hand (intent with that business's list + `other`, department, urgency; `data_domain/labels_<biz>.json`) and turned into
`data_domain/real_cs_eval.jsonl` (360 records, `cs/build_real_eval.py`). Every checkpoint and the teacher on it:

| | teacher | run 3 | run 5 | run 6 | run 7 |
|---|---|---|---|---|---|
| intent (per-business list) | 0.29 | 0.26 | 0.37 | 0.41 | **0.44** |
| department | 0.48 | - | - | 0.48 | 0.48 |
| urgency (exact) | ~0.2 | - | - | 0.22 | 0.22 |
| says `other` on in-scope messages | 0% | - | - | 70-85% | 70-85% |

Reading: **the 0.97 held-out numbers are for LLM-written text; on real posts the student is at 0.44 and the teacher at 0.29**
(it never says `other`, and its urgency reads "answer today" almost always). Runs 6/7 answer `other` for most genuine messages,
because their only real-text examples were out-of-scope posts. Synthetic data cannot fix this; real in-domain text with real
labels can, and the teacher's labels on real text are not usable for intent.

## Run 8: real Thai in-domain text (2026-09-27/28)

Three additions (`cs/label_cs8.py`, `cs/run8.sh`, 09:54 -> 02:37):

1. **932 more Pantip questions labelled by hand** (telecom 400, banking 400, insurance 132; `data_domain/labels_<biz>_extra.json`,
   `pantip_<biz>_extra.jsonl`), used as training items with intent / department / urgency one-hot and the labelled questions repeated 6x.
   The first 360 stay as the eval set.
2. **Intent lists extended with what real users ask** (`cs_questions.py`): telecom +5 (top_up_problem, device_or_equipment,
   change_account_details, value_added_service, complaint), banking +19 (the banking77 groups: card_delivery, card_not_working,
   transfer_problem, unrecognised_or_wrong_charge, refund_or_reversal, top_up_or_deposit_problem, identity_verification_kyc,
   account_suspended_or_fraud, app_or_login_problem, statement_or_document, loan_or_debt_restructuring, ...), insurance +1
   (policy_document_not_received). The 360 eval rows that were `other` only for lack of an option were re-labelled
   (`cs/fix_labels_run8.py`: banking `other` 61 -> 26 of 120).
3. **Real English question sets, rewritten in Thai** by Qwen3-4B (`cs/fetch_real_en.py`, `prep_cs2.sh`): banking77 (9,993 real
   banking questions, 77 labels -> `BANKING77_TO_CS`, 19,975 Thai rows) and insurance-qa (8,758 rows, topic -> `information_*`),
   kept when the teacher puts p >= 0.3 on the mapped intent or ranks it top-3.

Plus the run 7 records with intent targets rebuilt over the new lists (e-commerce capped at 30,000 texts). Labelling 3 h 20 min
(teacher gate kept banking77 13,390 / 19,975 and insurance-qa 5,858 / 8,758); 329,156 items (Pantip 24,713 = 7.5%, banking77 +
insurance-qa 54,900); train from run 3, 2 epochs, 21,238 updates, 11 h 49 min; temperatures 1.07 / 1.15 / 1.04.

**Real Pantip set, 360 hand-labelled questions the model never saw** (extended intent lists, `results/run*_realcs8.json`,
per-question split from `cs/build_real_eval.py` against `:8011` / `:8010`):

| | teacher | run 3 | run 7 | **run 8** |
|---|---|---|---|---|
| intent, telecom / banking / insurance | 0.45 / 0.30 / 0.16 | - | - | **0.58 / 0.56 / 0.52** |
| intent + department pooled, per business | - | 0.24-0.35 | 0.37-0.45 | **0.61-0.62** |
| department | 0.47 | - | - | **0.66** |
| urgency exact / MAE | 0.21 / 0.89 | 0.14-0.26 | 0.18-0.28 | **0.73 / 0.38** |
| overall (offline, pure student) | - | 0.258 | 0.332 | **0.656** |
| says `other` on in-scope messages | 0% | - | 70-85% | **7% / 13% / 30%** |
| ECE | - | 0.140 | 0.381 | 0.190 |

(Run 7's 0.44 intent in the previous section was on the old, shorter lists; on the same file as run 8 it is 0.33 pooled.)
The `other` reflex is gone, urgency now matches the hand labels (the teacher's does not), and intent doubled. Still 0.55, not
0.97: the remaining errors are real ambiguity (change_plan vs change_provider vs sign_up_for_plan on "ขอโปรถูก ๆ", apply_for_loan vs
loan_or_debt_restructuring) and the long tail of 45 banking intents with a handful of real examples each.

Held-out cs8 (5,445 texts): overall 0.958 (run 7 on the same file 0.895, because the b77 / insurance-qa rows are new to it:
0.573 / 0.637 -> 0.959 / 0.986); the synthetic sources stay at 0.95-0.99, wisesight 0.892 (-2.4), e-commerce 0.988 (-0.8). Public
set 0.780 (run 7 0.782): prachathai 0.896, sib200 0.672 (+4.0), wisesight 0.717 (+1.7), xnli 0.717 (-3.0). ECE 0.153.

**krathu-500** (`cs/build_krathu_eval.py`: 1,163 Pantip comments, POS / NEG / NEU balanced, no licence so eval only): teacher
0.660, run 3 0.650, run 5 0.674, run 6 0.638, run 7 0.619, run 8 0.623. Sentiment on real comments slid 5 points over the
call-center runs and the teacher itself is at 0.66 (its main error: "neutral" read as "negative", 144 / 382). Not a priority.

Served at `:8011` since 2026-09-28 08:30 (`laya-cascade:run8`; cascade with the teacher at threshold 0.7 gives intent 0.55 /
department 0.66 / urgency 0.73 on the real set). Published as
[Chatre7/laya-thai-callcenter](https://huggingface.co/Chatre7/laya-thai-callcenter) revision `e85ec494` (run 6 `6a4dc84a`).

**Cascade on real text** (`cascade.py --max-options 0` on the real set, `results/cascade8_real.json`, curves per question from
`results/cascade_curve.py`): student alone 0.656, teacher alone 0.328; at threshold 0.7 the cascade is 0.647 with 12% of the
questions sent to the teacher, which is wrong on 65-80% of what it receives (intent 0.20, urgency 0.32). On call-center
questions the teacher is not a useful fallback any more; a low-confidence answer should go to a person, not to OpenThai.
`:8011` keeps threshold 0.7 for the general questions (where the teacher still leads, public set 0.81 vs 0.78); for call-center
traffic set `CASCADE_THRESHOLD=0` or read `usage.cascade.reasons` and escalate those to a human.

**Next**: more hand-labelled real text is the only lever that moved the real-set number (932 rows: +0.3). Candidates: the rest of
the Pantip pools (2,192 telecom, 1,137 banking after sampling), the reviewed sheets in `System One/data/` once reviewed, and any
real ticket export from the call center.

## Run 9: 1,000 more hand-labelled Pantip rows (2026-09-28, in progress)

Second labelling batch (`cs/sample_pantip_extra2.py`, `data_domain/labels_{telecom,banking}_extra2.json`): telecom 500, banking 500,
sampled from the remaining pools (the insurance pool is exhausted); 1,932 real rows in total. Same recipe and data as run 8
otherwise (`label_cs8.py --pantip-sets extra,extra2`, `cs/run9.sh`, trained from run 3), so the difference measures what another
1,000 real rows buy on the 360-row real set. 356,189 items (Pantip 51,710 = 14.5%), 12 h 52 min, done 05:06.

**Nothing, on average.** Real Pantip set: run 8 0.656 -> run 9 0.650 overall; telecom 0.617 -> 0.654 (urgency 0.73 -> 0.80),
banking 0.608 -> 0.562, insurance 0.613 -> 0.542 (insurance got no new rows). Per business that is 240 decisions, so +-4 points is
within run-to-run noise; the honest reading is a plateau around 0.65 for this recipe. cs9 held-out: on the Pantip rows the model
never saw, run 9 scores 0.63-0.72 intent+department and 0.74-0.87 urgency (run 8's 0.80+ on the same rows is contaminated: the
split was re-cut, so it had trained on some of them). Public 0.776 (run 8 0.780), krathu-500 0.642 (+1.9). `:8011` stays on run 8.
`results/run9*.json`, `run8_cs9.json`, `cs9_manifest.json`, `compare_run9.py`.

Why more of the same did not help: the 45-intent banking list has several near-synonyms (apply_for_loan / loan_or_debt_restructuring,
make_transfer / transfer_problem, check_fees / check_card_annual_fee) and the labels for those are as noisy as the model; and real rows
are still 14% of the items, diluted by 300k synthetic ones. The next experiments are cheap: (1) a short second pass from run 8 on the
1,932 real rows only, (2) merge the near-synonym intents with the call center's list, (3) real tickets instead of forum posts.

## Run 10: a short second pass on the real rows only (2026-09-29)

`cs/run10.sh`: from run 8, only the 51,710 Pantip items of cs9 (`cs/filter_items.py`: the 1,932 hand-labelled rows, labelled
questions x6 + the teacher's shared questions), LR 1e-5 / 4e-5, 2 epochs, 2 h 6 min (vs 12 h for a full run).

| | run 8 | run 9 | **run 10** |
|---|---|---|---|
| real Pantip set, overall | 0.656 | 0.650 | **0.673** |
| intent + department: telecom / banking / insurance | 0.617 / **0.608** / **0.613** | 0.654 / 0.562 / 0.542 | 0.650 / 0.583 / 0.604 |
| urgency exact: telecom / banking / insurance | 0.725 / 0.783 / 0.725 | 0.800 / 0.792 / 0.742 | 0.783 / **0.808** / **0.800** |
| public set | **0.780** | 0.776 | 0.773 |
| synthetic held-out (cs9) | 0.968 | 0.954 | 0.963 |

The real rows were partly drowned: a cheap pass on them alone gains 1.7 points, almost all in urgency, while intent/department
stay within noise and the public set loses 0.7. Not deployed (`:8011` stays on run 8) pending the choice between better urgency
and banking/insurance routing. The recipe itself is the useful result: new real data can be added to the served model in two hours.
`results/run10*.json`, `compare_run10.py`.

## laya 0.3.20 (2026-09-28)

Upstream moved 0.3.5 -> 0.3.20 in four days (350 commits, mostly outside PRs): `predict_long` (states past the context window),
calibrated confidence next to the raw one, batching by encoded length, the TileLang fast path (`agent.accelerate()`), an ONNX
agent, `laya serve`, `decide()` with pydantic schemas, MCP / LangChain, revision pinning, fixes for a tokenizer race across threads
and for the decision-head init. Our fork had touched nothing under `laya/`, so the merge (branch `thai`, commit `412ef72`) only
conflicted in `.gitignore`. Checked on the box in a separate clone (`cs/check_0320.sh`, `~/laya-0320`, the old code keeps serving
`:8011`): run 8 evaluated with 0.3.20 reproduces every number of `run8.sh` to four decimals on both the real Pantip set (0.6565 /
0.5590 / 0.1902) and the public set (0.7796 / 0.3802 / 0.1534) (`results/compare_0320.py`). The upstream test suite needs
`pytest`, which the training image lacks, and most of its files are scripts that call `sys.exit` at import, so pytest cannot
collect them; not pursued.

**TileLang fast path: works, +20% on the A2, not deployed.** `agent.accelerate()` JIT-compiles CUDA kernels, so it needs a CUDA
*toolkit*; the OpenThai base image has runtime libraries only. `Dockerfile.0320` adds g++ and the pip wheels `nvidia-cuda-nvcc`,
`nvidia-cuda-cccl`, `nvidia-nvvm`, `nvidia-nvjitlink`, all pinned to 13.0 to match the CUDA 13.0 runtime torch ships with (every
unpinned piece resolved to 13.4 and broke a different stage: no nvcc, no host compiler, `nv/target` missing, "compiler and headers
incompatible", `ptxas: Unsupported .version 9.4`). `bench_fast.py` (run 8, 120 real Pantip states, intent + department + urgency,
A2, bf16, `results/bench_fast8.json`):

| | stock 0.3.20 | fast path |
|---|---|---|
| one record, 3 questions, p50 / p95 | 126 / 142 ms | **102 / 118 ms** |
| `predict_batch`, 16 records, p50 / p95 | 2.13 / 2.16 s | 1.85 / 10.4 s (graph capture per new shape) |
| first-use compile | - | 61 s |
| argmax agreement with the stock path | - | 359 / 360 choices, scores within 0.04 |

20% at batch 1 and unstable at batch 16 on this GPU: the fused kernels save launch overhead, and the A2 is compute-bound on our
~800-token intent sequences. The `:8011` server (own batching + `torch.compile`, ~50 ms per call) stays as it is; the fast path
is a documented option for a bigger GPU. ONNX (`ONNXAgent`) targets CPU deployment and was not measured.

## A second domain: web agent, run web1 (2026-09-29)

On OpenThai's browser demo (`System One/demo/browser_agent.py`, 9 single-step Thai tasks on Wikipedia and the-internet.herokuapp)
the call-center run 8 picked the right element 5/9 times, the teacher 7/9. `web/` trains a separate model for this, from run 3
(general Thai distillation) rather than the call-center model:

- `web/prep_m2w.py`: Multimodal-Mind2Web (osunlp, OpenRAIL), text columns only (range reads over `hf://`, no screenshots) ->
  records in the demo's element format (`[i] <Thai role> (<role>) "<name>"`): the positive element plus 3-19 random
  interactive-looking negatives (mean 12 options), last 3 previous actions. Train 7,362 steps; test_task 1,257,
  test_website 975, test_domain 3,838.
- `web/translate_tasks.py`: the 2,022 task instructions to Thai with Qwen3-4B on vLLM (2,011 kept, 4 min).
- `web/label_web.py`: two choice questions per step, `target` and `operation` (CLICK / TYPE / SELECT), on the English and on the
  Thai task, English or Thai question wording; human labels one-hot with smoothing 0.1, no teacher. 29,316 items.
- `web/run_web1.sh`: 3 epochs, 2 h 12 min on GPU 1 -> `out/laya-th-web1`.

Mind2Web test splits (`web/eval_web.py`, `results/{web1,run3}_m2w_split.json`), Thai task / English task:

| | run 3 (no web data) | **web1** | random / always CLICK |
|---|---|---|---|
| element, test_task | 0.127 / 0.146 | **0.668 / 0.716** | 0.10 |
| element, test_website | 0.153 / 0.168 | **0.699 / 0.735** | 0.11 |
| element, test_domain | 0.139 / 0.178 | **0.596 / 0.661** | 0.11 |
| operation, test_task / website / domain (Thai) | 0.29 / 0.26 / 0.31 | 0.90 / 0.90 / 0.89 | 0.83 / 0.80 / 0.83 |
| element and operation both right, test_domain | 0.046 / 0.050 | **0.546 / 0.612** | |

Unseen websites cost nothing and unseen domains 7 points; the translated Thai tasks trail English by 4-7. The operation head is
only 6-10 points over always-CLICK. Not comparable to published Mind2Web numbers, which rank far more candidates per page.

Browser demo, 9 tasks. `--history` (new) sends the scenario's earlier actions as `state.previous`, as in training:

| | no history | `--history` | ms per step |
|---|---|---|---|
| web1, laya only (`:8014`, temporary) | 5/9 | 6/9 | 100-300 |
| run 8 cascade (`:8011`) | 5/9 | 6/9 | 130-1,900 |
| teacher (`:8010`) | 7/9 | 8/9 | 110-1,500 |

With history web1 clicks Login after the password (without it, it picks the password box again: Mind2Web textboxes show no typed
value, so the model learned to read progress from `previous`). Its remaining misses are one error on Wikipedia: for "type
กรุงเทพมหานคร in the search box" it picks the Search *button* while its own operation head says TYPE (p 0.89), and the
following two tasks then run on the wrong page.

`--joint` (new) picks the element maximising p(element) x p(operation its role implies). It does not rescue web1 here: web1
gives the Search button 0.72 and the search box 0.04, whatever the wording (Thai or English, "type" or "search"); with the button
removed from the list the box gets 0.76. The training data does not teach this: among the 71 train steps offering both a
search-like box and a search-like button, with no TYPE just before, the target is the box 28 times and the button 5
(`web/search_bias.py`). More likely a role prior (textbox is the target in only 13% of train steps). Run 8 through the cascade
gains the dropdown task (6/9 -> 7/9), where its element head preferred a link but its operation head said SELECT (0.83). The
teacher stays 8/9.

| `--history --joint` | web1 | run 8 cascade | teacher |
|---|---|---|---|
| element | 6/9 | 7/9 | 8/9 |

Not served; `:8011` stays call-center run 8.

### Plan once, decide narrow: the Laya Ultrafast split (2026-09-29)

[Laya Ultrafast](https://github.com/ipenywis/laya-ultrafast) (madewithlaya.com) makes one LLM call per task (goal -> field
values, submit, target, finish condition) and asks laya only narrow questions per step. `System One/demo/form_agent.py` does the
same with Qwen3-4B-FP8 on vLLM as planner (plans again after opening a menu, since the form appears only then), any
`/v1/systemone` endpoint as decider, Playwright acting, and success checked on the page, not by the model. Eight Thai tasks: four on
a mock Thai CS portal (`demo/sites/crm.html`: open a case with phone / category / channel / urgency / SMS, change a billing
address, find a customer through the autocomplete), Thai and English Wikipedia articles, the-internet login and dropdown.

| decider | tasks passed | ms per decision |
|---|---|---|
| OpenThai-SystemOne (`:8010`) | **8/8** | ~1,060 (Wikipedia pages: 100+ elements) |
| laya run 8, student only | 0/8 | ~130 |

The split works: with the teacher deciding, every task passes, including all four call-center forms. laya does not. Replaying
the teacher's 31 narrow decisions (`web/probe_form.py`, the teacher is right on all of them by construction) against every
checkpoint:

| | field for a value | option matching a value | submit button | suggestion | total |
|---|---|---|---|---|---|
| laya-multilingual (upstream base) | 8/17 | 1/7 | 1/4 | 1/3 | 11/31 |
| run 3 | 11/17 | 3/7 | 1/4 | 1/3 | 16/31 |
| run 8 | 9/17 | 2/7 | 4/4 | 1/3 | 16/31 |
| run 10 | 12/17 | 3/7 | 4/4 | 1/3 | 20/31 |
| web1 | 8/17 | 4/7 | 4/4 | 1/3 | 17/31 |

"Which option matches "สูง"" -> ต่ำ; "matches "Option 2"" -> Option 1; "which box takes "tomsmith" (Username)" -> Password. Copying
a string from the question to the matching option is what our checkpoints and the upstream base cannot do in Thai, and it is the
whole job in this split. Two fixes: match strings in code first and ask laya only when several options fit (done, below), and
train on synthetic matching questions (gold by construction, any number, Thai and English; not done).

**String matching first** (`form_agent.py`, `--no-match` turns it off). `string_match`: equal after normalising (case, spaces,
punctuation) wins; else exactly one option containing the query (or contained in it) wins; several such are ranked by
`SequenceMatcher` and the best wins only by a 0.15 lead, otherwise the model chooses **among those only**. A field whose
name matches nothing goes to the one select/radio offering the value. The target step is skipped when the visible page heading
(h1 first) already matches it. Same eight tasks:

| decider | no matching | **matching first** | string matches / model calls | ms per model call |
|---|---|---|---|---|
| laya run 8, student only (`:8014`) | 0/8 | **8/8** | 34 / 13 | 109 |
| run 8 cascade (`:8011`) | - | **8/8** | 34 / 12 | 1,331 |
| teacher (`:8010`) | 8/8 | **8/8** | 34 / 11 | 1,518 |

Code settles about three decisions in four; laya's remaining ones are the fuzzy ones (which of 11 autocomplete rows, which of
4 "Chiang Mai" links, which menu to open on a 114-element page). It still errs there (on Wikipedia it takes "Search for pages
containing ..." instead of the article) but the next step recovers. The models' own "done" answer is not usable: laya said done
on 5 of 8 passed tasks, the teacher on 1. Caveats: eight tasks, and the portal is ours, its labels read by the planner.
`results/form_agent_match.txt`, `results/form_agent_nomatch.txt`.

## Next best action: banking, telecom, insurance (2026-09-29)

What should the agent do next, from the message plus two facts the agent's screen knows (identity verified? how many
contacts about this?). `cs/nba_actions.py`: DRAFT playbooks, 22 banking actions (block card, fraud freeze, open dispute,
trace a transaction, ATM case, sales referral, debt relief, retention offer, ...), 19 telecom (suspend SIM, troubleshoot,
network ticket, billing review, top-up case, plan sales, retention offer, SIM / feature / device service, ...), 19 insurance
(claim intake, claim status, settlement, appeal, coverage check, policy change, sales quote, renewal, retention offer, ...),
each with escalate / ask for clarification / out of scope; the context rule (`apply_context`: unverified -> verify first,
except emergencies; third unresolved contact -> escalate) and a table from each business's intents to the usual action.
Eval: the 120 real Pantip rows per business of the real-text set, hand-labelled with the best action for a verified first
contact (plus acceptable alternatives, `data_domain/nba_labels_<business>_eval.json`), x 4 contexts through the rule = 480
decisions per business (`cs/label_nba.py --eval-only` writes it, `cs/eval_nba.py` scores it, `cs/nba_eval_all.sh` runs all).
Top-1 counts the best or an acceptable action; top-3 the best action among the three highest.

| top-1 / top-3 | banking | telecom | insurance |
|---|---|---|---|
| laya run 8 asked the NBA question directly | 0.271 / 0.548 | 0.392 / 0.506 | 0.281 / 0.602 |
| teacher asked with the context in the state | 0.192 / 0.335 | 0.508 / 0.610 | 0.237 / 0.365 |
| teacher on the message only + context rule | 0.442 / 0.523 | 0.573 / 0.637 | 0.138 / 0.325 |
| **laya run 8 intent -> table -> context rule** | **0.690 / 0.808** | **0.667 / 0.733** | **0.656 / 0.754** |
| most frequent gold action, always | 0.24 | 0.24 | 0.43 (out of scope) |

(Rule scores are the summed intent probability per action; ranking intents and taking their actions in order gave 0.696 /
0.756 on banking.) The teacher ignores the context fields (it answers "verify identity" for customers marked verified); on the
message alone it is uneven: fair on telecom, and on insurance it files a claim for nearly everything, including the 52 posts
that are not insurance questions at all (phone warranties, loans). Distilling it would teach laya something worse than the
table, so run nba1 (`cs/run_nba1.sh`, teacher labels spread over the contexts by the rule, replay of cs9/Pantip items) was
**not trained**. What works today is the existing intent model plus the playbooks: `cs/nba_demo.py --business ...` asks `:8011`
for intent / urgency / frustration and scores each action as the summed probability of the intents that map to it (after the
context rule), ~0.3 s. The rule's main misses: telecom posts about switching operators, labelled as a sales lead for the
operator receiving them while the table sends `change_provider` to retention (the post rarely says which side we are);
insurance and banking questions the intent model calls `other`.
Caveat: the tables and the hand labels are both mine, so these numbers flatter the tables; the real test is a playbook written
by the call center and actions agents took on real tickets. A learned NBA model needs those (ticket -> action that resolved
it); until then, better intent accuracy is better NBA.

## Run 11: short chat versions, human frustration labels, hard examples (2026-09-30)

Three fixes aimed at the intent errors behind the next-best-action misses, and at frustration over-scoring, all on the real rows:

- **Short chat versions.** Real callers type one or two sentences; our only real texts are long forum posts. `cs/shorten_pantip.py`
  rewrites each of the 2,292 hand-labelled Pantip posts as a chat message to the company (Qwen3-4B-FP8 on vLLM, two styles,
  4,560 kept of 4,584, 16 min); the human intent / department / urgency carry over. The 360 eval posts' versions (714) are a
  new **short** eval set; the rest (3,846) train.
- **Human frustration.** 660 rows hand-labelled 0 / 1 / 2 (`data_domain/frustration_real_eval.json` for the 360 eval posts,
  `frustration_train.json` for 300 training posts). The teacher's frustration targets are dropped on the Pantip rows.
- **Hard examples.** In-scope rows run 8 calls `other` (132 of 1,534 long, 299 of 3,211 short) get their intent item 3 more times.

`cs/label_cs11.py`: 67,716 real items + 20,000 replayed synthetic cs9 items; `cs/run11.sh`: second pass from run 8, 2 epochs,
LR 1e-5 / 4e-5, 3 h 26 min. `cs/eval_real.py` scores long and short, false-other (in-scope called `other`) and frustration.

| | run 8 | run 10 | **run 11** |
|---|---|---|---|
| long (360): intent / department / urgency | 0.575 / 0.647 / 0.747 | 0.547 / 0.681 / 0.797 | 0.578 / **0.700** / **0.797** |
| long: false-other (telecom / banking / insurance) | 9/107, 12/94, 21/64 = 16% | 1, 5, 14 = **7.5%** | 4, 7, 20 = 12% |
| long: frustration exact / MAE / mean predicted vs true | 0.50 / 0.57 / 0.93 vs 0.46 | 0.50 / 0.57 / 0.96 vs 0.46 | **0.77 / 0.30 / 0.52 vs 0.46** |
| short (714): intent / department / urgency | 0.520 / 0.591 / 0.601 | 0.496 / 0.653 / 0.721 | **0.550 / 0.691 / 0.793** |
| short: false-other | 12% | **7.8%** | 13% |
| next best action, rule top-1: banking / telecom / insurance | 0.690 / 0.667 / 0.656 | - | 0.685 / **0.696** / **0.710** |
| cs9 held-out / public set | 0.968 / 0.780 | 0.963 / 0.774 | 0.965 / 0.773 |

What moved: frustration is fixed (the model over-scored calm messages by half a point; now exact 0.77 and no bias),
department and urgency gain 5 points on the long posts and 10-19 on the short ones, next-best-action gains on telecom and
insurance. What did not: intent itself is flat on the long posts (+3 on the short ones), and false-other only fell from 16% to
12% on long, not at all on short, despite the hard examples; run 10 (Pantip-only pass) still has the lowest false-other but
the lowest `other` recall. Insurance stays the weak business (20/64 in-scope long posts called `other`). Public set -0.7.
`results/real_baselines.json`, `real11.json`, `nba_*_run11.json`, `run11*.json`, `run11.log`. Not served yet: `:8011` stays
on run 8 pending the choice.

## Run 12: Google Play reviews (2026-09-30/10-01)

The Hub has no Thai call-center text (`cs/hf_thai_search.py`: 254 Thai-tagged datasets, 329 from Thai organisations; the
closest are a 33 GB role-played call-center speech set, CC-BY-NC, and translated scam dialogues). Google Play reviews of Thai
bank / telecom / insurance apps are public, short (63 chars on average) and in the customer's own words: `cs/fetch_play_reviews.py`
fetched 20,867 (banking 12.9k incl. wallets and a consumer-loan app, insurance 4.6k, telecom 3.3k); 300 held out and
hand-labelled (`data_domain/play_labels.json`: 142 are praise or no request = `other`, 83 app/login problems).

Labelling (`cs/label_play.py`): the teacher answers every question and run 11 the intent; a review is kept when the teacher's
probability of the student's intent is >= 0.2. Only **1,234 of 12,000** pass (strict top-1 agreement 10%): the teacher invents
intents for praise ("ดีมาก" -> apply_for_card 0.96) and the student calls many real problems `other`. 11,106 review items +
40,000 replayed run 11 items, a pass from run 11 (`cs/run12.sh`, 2 epochs, 2 h). `cs/eval_play.py` scores the 300.

| | run 11 | **run 12** |
|---|---|---|
| reviews (300): intent / department / urgency | 0.557 / 0.623 / 0.467 | **0.633** / 0.620 / 0.253 |
| reviews: false-other (in-scope called other) | 115/158 = 73% | **58/158 = 37%** |
| long (360): intent / department / urgency | 0.578 / 0.700 / 0.797 | 0.556 / 0.686 / 0.797 |
| long: false-other | 12% | **8.3%** |
| short (714): intent / department / urgency | 0.550 / 0.691 / 0.793 | 0.555 / 0.693 / 0.763 |
| short: false-other | 13% | **10.6%** |
| next best action, rule top-1: banking / telecom / insurance | 0.685 / 0.696 / 0.710 | 0.675 / 0.683 / **0.727** |
| cs9 held-out / public set | 0.965 / 0.773 | 0.962 / 0.769 |

The reviews teach what they contain: short app complaints. On them intent +8 and false-other halves; false-other also falls
on the long and short Pantip sets. The costs: banking intent on the long posts -6 (0.533 -> 0.475), urgency on the reviews
collapses (the teacher's urgency targets on the review items say "urgent" for app complaints that the hand labels call
"today"), everything else within a point. Run 11 remains the better all-round checkpoint; run 12 only if short app-style
messages are the target. The useful outcome is the recipe and the 300-review eval set, and the finding that the teacher cannot
label reviews. `results/real12.json`, `play12.json`, `nba_*_run12.json`, `run12*.json`, `run12.log`.

## Run 13: the reviews labelled by an LLM (2026-10-01)

Same reviews, a different labeller: Qwen3-8B-FP8 on vLLM with the business's intent list, the departments and the urgency
scale in the prompt and a JSON-schema-constrained answer (`cs/label_play_llm.py`). Against the 300 hand labels it scores
intent 0.74 / department 0.72 / urgency 0.78 (0.71 / 0.67 / 0.73 before one prompt revision that states the convention for
app problems, made after seeing the first errors on these same 300, so the 300 are not a clean test of the labeller). It labels
all 20,567 reviews in 90 min (13,137 `other`, 4,228 app/login problems) versus 1,234 usable rows from teacher+student agreement.
`cs/label_cs13.py` caps each (business, intent) at 1,500 rows -> 9,202 reviews, 41,510 items, + 50,000 replayed run 11 items;
a pass from run 11 (`cs/run13.sh`, 2 epochs, 4 h 8 min).

| | run 11 | run 12 (teacher+student) | **run 13 (LLM labels)** |
|---|---|---|---|
| reviews (300): intent / department / urgency | 0.557 / 0.623 / 0.467 | 0.633 / 0.620 / 0.253 | **0.737 / 0.740 / 0.807** |
| reviews: false-other | 73% | 37% | **28%** |
| long (360): intent / department / urgency | **0.578 / 0.700** / 0.797 | 0.556 / 0.686 / 0.797 | 0.561 / 0.683 / **0.822** |
| long: false-other | 12% | **8.3%** | 9.8% |
| short (714): intent / department / urgency | 0.550 / 0.691 / 0.793 | 0.555 / 0.693 / 0.763 | 0.553 / **0.702 / 0.800** |
| short: false-other | 13% | 10.6% | **10.4%** |
| next best action, rule top-1: banking / telecom / insurance | **0.685** / 0.696 / 0.710 | 0.675 / 0.683 / **0.727** | 0.665 / **0.721** / 0.721 |
| cs9 held-out / public set | **0.965 / 0.773** | 0.962 / 0.769 | 0.955 / 0.757 |

The student reaches the labeller's own accuracy on the reviews (0.737 vs 0.74) at 1/100 of the cost per message, so on this
kind of text the ceiling is now the labels, not the model. The Pantip sets are flat (urgency +2.5, banking intent on long posts
-4), and the price is 1.6 points on the public set and 1 on cs9: 41k items of one narrow register pull the model that way.
Which to serve depends on the traffic: short app/chat-style messages -> run 13; forum-length or mixed -> run 11.
`results/play13.json`, `real13.json`, `nba_*_run13.json`, `run13*.json`, `run13.log`.

## Run 14: LLM labels on the forum posts, kept where the student agrees (2026-10-01/02)

Can the LLM labeller also extend the Pantip data? `cs/label_pantip_llm.py --eval`: on the 360 hand-labelled posts Qwen3-8B
scores intent 0.57 / department 0.65 / urgency 0.79, the same as run 11 (0.58 / 0.70 / 0.80), so its labels alone teach
nothing. But the two err on different posts (`cs/agree_check.py`): they agree on the intent of 50% of posts and are right on
80% of those (department: agree 57%, right 88%; urgency: agree 76%, right 90%); at least one is right on 75%.

`cs/label_cs14.py`: the 3,558 posts without hand labels (2,634 Pantip, 924 wisesight) labelled by the LLM, each question kept
only where run 11 agrees (intent 2,213, department 2,047, urgency 2,163); the LLM-labelled reviews capped at 800 per
(business, intent) instead of 1,500 (6,376 reviews); 60,000 replayed run 11 items. 41,482 new items, a pass from run 11
(`cs/run14.sh`, 2 epochs, 4 h 43 min).

| | run 11 | run 13 | **run 14** |
|---|---|---|---|
| reviews (300): intent / department / urgency | 0.557 / 0.623 / 0.467 | 0.737 / 0.740 / 0.807 | **0.743** / 0.737 / **0.820** |
| long (360): intent / department / urgency | **0.578** / 0.700 / 0.797 | 0.561 / 0.683 / **0.822** | 0.561 / **0.722** / 0.806 |
| long intent: telecom / banking / insurance | 0.558 / 0.533 / **0.642** | 0.575 / 0.492 / 0.617 | 0.575 / **0.550** / 0.558 |
| short (714): intent / department / urgency | 0.550 / 0.691 / 0.793 | 0.553 / 0.702 / **0.800** | **0.560 / 0.712** / 0.788 |
| false-other: reviews / long / short | 73% / 12% / 13% | 28% / **9.8%** / 10.4% | **27%** / 13% / **9.7%** |
| next best action, rule top-1: banking / telecom / insurance | 0.685 / 0.696 / 0.710 | 0.665 / **0.721 / 0.721** | **0.713** / 0.652 / 0.662 |
| cs9 held-out / public set | **0.965 / 0.773** | 0.955 / 0.757 | 0.957 / 0.763 |

Run 14 keeps run 13's gains on the reviews, has the best department everywhere and gives back part of the public-set loss, but
it is not a clear step past run 13: on 360 posts one standard error is ~2.6 points and the per-business moves (banking up,
insurance down) are inside that. Agreement labels are mostly posts the student already gets right, so they steady the model
more than they teach it. Averaged over the three real-text sets: intent 0.562 / 0.617 / **0.621**, department 0.671 / 0.708 /
**0.724**, urgency 0.686 / **0.810** / 0.805 (run 11 / 13 / 14). Intent on forum posts has been ~0.56-0.58 since run 8; what
moves it now is better labels (call-center staff, or a taxonomy with fewer overlapping intents), not more of this.
`results/real14.json`, `play14.json`, `nba_*_run14.json`, `run14*.json`, `run14.log`.

**Served since 2026-10-02: `:8011` runs run 14** (`docker-compose.cascade.yml` mounts `out/laya-th-run14`; roll back by mounting `out/laya-th-run8`). The cascade threshold is unchanged (0.7), although on real call-center text the teacher fallback does not help.

## Merging the intents (2026-10-02)

The 116 fine intents come from the Bitext taxonomies, not from a Thai call center, and several overlap (transfer / transfer
problem / cancel transfer; change plan / sign up / change provider; six `information_*`). `cs/intent_groups.py` is a DRAFT
merge into the groups an agent would handle differently: telecom 31 -> 11, banking 45 -> 15, insurance 40 -> 14 (+ `other`).
Nothing is retrained: the model answers the fine question and the probabilities are summed per group (`group_probs`).
`cs/eval_groups.py` on the 1,374 hand-labelled real texts (Pantip long 360, short 714, reviews 300):

| run 14 | fine intent | **merged group** | group in top 3 |
|---|---|---|---|
| telecom (408) | 0.598 | **0.672** | 0.821 |
| banking (533) | 0.612 | **0.696** | 0.773 |
| insurance (433) | 0.587 | **0.654** | 0.734 |
| all | 0.600 | **0.675** | 0.775 |
| run 8, all | 0.544 | 0.604 | 0.702 |

+7.5 points for free, less than the intent -> action table suggested, because most of what is left is not confusion between
neighbouring intents but the in-scope / `other` boundary (banking: app or login problem called `other` 26 times; insurance:
general information called `other` 17, `other` called buy/quote 12), which no merge fixes. One taxonomy artefact shows too:
telecom `payment -> top_up` 18 times, where the eval labels (written before `top_up_problem` existed) say
`check_mobile_payments` for top-up mistakes. `cs/INTENT_GROUPS.md` is the sheet for the call-center team (groups in Thai, the
fine intents inside each, accuracy, remaining confusions); `results/groups.json`.

## Test desk for the call-center team (`ui/`, port 8020)

`http://172.18.72.145:8020`: a page in Thai where an agent types a customer message (or picks a sample), chooses the business
and the context (verified, repeat contact) and gets a "case slip": merged intent group with its probability and the runners-up,
department, urgency, customer mood, next best actions, from the served model at `:8011` (`ui/app.py`, FastAPI, no GPU;
`ui/index.html`, one file; `ui/deploy.sh` runs it from the `laya-cascade` image). Under the slip the agent marks the answer
right or wrong and, if wrong, picks the right group and department; each click appends one line to `ui_data/feedback.jsonl`
(message, context, the model's answer, the correction, tester, time). Nothing is stored on analyze alone, and the page asks
testers to remove names, phone and account numbers first. The feedback file is the human-labelled call-center text this
project lacks; it stays on the box and is not committed.

## Mood against the customers' own stars, and sarcasm (2026-10-02)

`cs/eval_stars.py`, no training: 3,000 Google Play reviews with the star rating the reviewer gave, and the 3,300 sentences
of `Tippawan/thai-ambiguous-sentiment` (sarcasm, indirect praise, neutral facts; pos / neg / neu labels, provenance and licence
not stated on the Hub, so a test set only). `results/stars.json`.

| run 14 on reviews | 1 star | 2 | 3 | 4 | 5 stars |
|---|---|---|---|---|---|
| `sentiment` = negative | 0.89 | 0.77 | 0.65 | 0.29 | 0.15 |
| `sentiment` = positive | 0.01 | 0.03 | 0.07 | 0.30 | 0.58 |
| mean `frustration` (0-2) | 1.29 | 1.10 | 1.06 | 0.82 | 0.57 |

1-2 stars called negative or 4-5 stars called positive: 0.72 (run 8: 0.63); the opposite polarity on 8.6%; the rest neutral.
Frustration falls with every star and separates 1-2 from 4-5 stars with AUC 0.89. The 15% of 5-star reviews called negative
are mostly real: people give five stars and write "เข้าแอปไม่ได้". Sarcasm is a different story: 0.49 on the ambiguous set,
19/100 on the sentences marked sarcasm; "ระบบทำงานได้สมบูรณ์แบบมากจ้า ค้างไปแค่ 10 รอบเอง" reads as praise (negative ->
positive 529 of 1,100), and indirect praise reads as neutral or negative. Neither run was trained on anything like it.

## Run 15: sarcasm and star-labelled mood - works, and breaks "neutral" (2026-10-02)

A pass from run 14 (`cs/run15.sh`, 76,589 items = 16,589 new + 60,000 replay), `sentiment` and `frustration` only:

- **Composed contrast sets** (`cs/gen_sarcasm.py`): an LLM asked for sarcasm writes complaints with a smile, so Qwen3-8B writes
  only the *events* (something that went wrong, the same thing played down, something that went well, an indirect compliment)
  and fixed Thai praise / complaint frames are put around them: praise + bad event = sarcastic, praise + good event = sincere,
  and so on. 5,390 messages after a filter, 10% held out.
- **Mined reviews** (`cs/mine_sarcasm.py`, `data_domain/sarcasm_mined.json`): the 111 low-star reviews run 14 called positive,
  read by hand: 13 sarcasm, 69 plain complaints, 17 praise with a low star, 12 unclear. The first two go in, x6.
- **Stars as labels**: 3,000 reviews of 1-2 stars as negative and 3,000 of 4-5 stars as positive, skipping those run 14 reads
  as the opposite polarity (five stars and a complaint is common).

| | run 14 | run 15 |
|---|---|---|
| reviews: 1-2 stars negative or 4-5 stars positive | 0.720 | **0.879** |
| reviews: the opposite polarity | 0.086 | 0.113 |
| frustration, 1-2 vs 4-5 stars (AUC) | 0.886 | 0.905 |
| held-out composed: sarcastic / sincere / indirect praise / plain negative | 0.62 / 0.80 / 0.33 / 0.97 | 0.98 / 0.99 / 0.98 / 1.00 |
| ambiguous set: sarcasm (100) / indirect praise (1,000) | 19 / 254 | 48 / 925 |
| ambiguous set: neutral facts (1,000) / Neutral-Fact (50) / Mixed (50) | 873 / 42 / 20 | **158 / 5 / 1** |
| ambiguous set, all 3,300 | 0.492 | 0.506 |
| Wisesight sentiment: public set / cs9 held-out | 0.67 / 0.88 | **0.55 / 0.82** |
| public set / cs9 held-out, overall | 0.763 / 0.957 | 0.744 / 0.949 |
| real posts, long: intent / department / urgency | 0.561 / 0.722 / 0.806 | 0.586 / 0.728 / 0.814 |
| reviews (300 hand labels): intent / department / urgency | 0.743 / 0.737 / 0.820 | 0.730 / 0.750 / 0.827 |

The sarcasm half works on text it has not seen (the ambiguous set is not ours). But every new sentiment item was positive or
negative, and the model stopped answering "neutral": 1% of reviews are called neutral where run 14 said 9-40%, a neutral fact
is now called positive (749 of 1,100), and Wisesight loses 12 points. A 3-star "เปลี่ยนอีเมลไม่ได้" should be negative, a
"สอบถามครับ แอปตัวเก่าจะไม่ใช้แล้วหรอ" should not. Mean frustration also dropped at every star (1.29 -> 1.05 at one star) while
its ranking improved. **Run 15 is not served.** Triage is unchanged within noise.

## Run 16: the same pass with neutral and question examples (2026-10-02/03)

`cs/run16.sh`: run 15's items plus the Wisesight training split under the `sentiment` question (3,000 neutral, 462 question
x2, 1,000 positive, 1,000 negative; the 601 texts that are also in the eval sets left out), 82,513 items, again from run 14.
`results/*16*`.

| | run 14 | run 15 | run 16 |
|---|---|---|---|
| reviews: 1-2 stars negative or 4-5 stars positive / the opposite | 0.720 / 0.086 | 0.879 / 0.113 | 0.861 / 0.107 |
| frustration, 1-2 vs 4-5 stars (AUC) | 0.886 | 0.905 | 0.910 |
| held-out composed: sarcastic / sincere / indirect praise / plain negative | 0.62 / 0.80 / 0.33 / 0.97 | 0.98 / 0.99 / 0.98 / 1.00 | 0.98 / 0.99 / 0.98 / 1.00 |
| ambiguous set: sarcasm (100) / indirect praise (1,000) | 19 / 254 | 48 / 925 | 43 / 889 |
| ambiguous set: neutral facts (1,000) / Neutral-Fact (50) / Mixed (50) | 873 / 42 / 20 | 158 / 5 / 1 | 386 / 16 / 0 |
| ambiguous set, all 3,300 | 0.492 | 0.506 | **0.552** |
| Wisesight sentiment: public set / cs9 held-out | 0.67 / 0.88 | 0.55 / 0.82 | 0.69 / 0.88 |
| public set / cs9 held-out, overall | 0.763 / 0.957 | 0.744 / 0.949 | 0.760 / 0.957 |
| real posts, long: intent / department / urgency (false-other) | 0.561 / 0.722 / 0.806 (12.4%) | 0.586 / 0.728 / 0.814 (12.4%) | **0.614** / 0.722 / 0.811 (9.1%) |
| real posts, short: intent / department / urgency | 0.560 / 0.712 / 0.788 | 0.553 / 0.710 / 0.800 | 0.560 / 0.695 / 0.809 |
| reviews (300 hand labels): intent / department / urgency | 0.743 / 0.737 / 0.820 | 0.730 / 0.750 / 0.827 | 0.723 / 0.747 / 0.840 |

Wisesight is back where it was and the sarcasm gains stay (43/100 against 19), so the two were not in conflict, only
the mix was. "Neutral" is half-recovered: a neutral fact in the ambiguous set is still called positive 500 times of 1,100
(run 14: 2), the mixed sentences never get "neutral", and reviews are called neutral 1-6% of the time where run 14 said
9-40% (most of those were complaints, so part of that is right). Intent on long posts is the best so far, 0.614, mostly
from fewer false `other`.

**Served since 2026-10-03: `:8011` runs run 16** (roll back by mounting `out/laya-th-run14`). Spot check through the server:
"ระบบทำงานได้สมบูรณ์แบบมากจ้า ค้างไปแค่ 10 รอบเอง" -> negative 0.90 (run 14: positive), "สอบถามครับ แอปตัวเก่าจะไม่ใช้แล้วหรอครับ"
-> question 0.82, "วันนี้ไปต่อบัตรที่สาขามา รอประมาณ 20 นาที" -> neutral 0.78, 30 ms each.

## Run 17: more weight on neutral - a trade, not served (2026-10-03/04)

`cs/run17.sh`: run 16 with 6,000 Wisesight neutral x2 and question x3 (91,975 items), again from run 14. `results/*17*`.

| | run 16 (served) | run 17 |
|---|---|---|
| ambiguous set: neutral facts (1,000) / Neutral-Fact (50) / Mixed (50) | 386 / 16 / 0 | **603 / 24 / 5** |
| ambiguous set: sarcasm (100) / indirect praise (1,000) | 43 / 889 | 42 / 680 |
| ambiguous set, all 3,300 | 0.552 | 0.555 |
| Wisesight sentiment: public set / cs9 held-out | 0.69 / 0.88 | **0.71** / 0.87 |
| public set / cs9 held-out, overall | 0.760 / 0.957 | 0.770 / 0.955 |
| reviews: 1-2 stars negative or 4-5 stars positive / the opposite | 0.861 / 0.107 | 0.840 / 0.101 |
| frustration, 1-2 vs 4-5 stars (AUC) | 0.910 | 0.913 |
| held-out composed: sarcastic / sincere / indirect praise / plain negative | 0.98 / 0.99 / 0.98 / 1.00 | 0.99 / 0.98 / 0.98 / 1.00 |
| real posts, long: intent / department / urgency (false-other) | **0.614 / 0.722 / 0.811** (9.1%) | 0.578 / 0.700 / 0.794 (13.2%) |
| real posts, short: intent / department / urgency | 0.560 / 0.695 / 0.809 | 0.559 / 0.700 / 0.793 |
| reviews (300 hand labels): intent / department / urgency | 0.723 / 0.747 / 0.840 | 0.730 / 0.750 / 0.830 |

More neutral examples buy more "neutral": 603 of 1,000 neutral facts (run 14: 873), Wisesight at its best. The price is
indirect praise back to neutral (889 -> 680) and, less expected, the long forum posts: intent, department and urgency all
drop and false `other` goes from 9% to 13%. 360 posts is a small set (13 posts between the two runs), but the three
questions move together, and 12,900 mood items against 60,000 replayed ones is the largest share of new single-question
items so far. Triage is what the desk uses; run 16 stays at `:8011`. "Neutral" on general text is the known weak spot of the
served model.

## Someone else's test: hagsmand1/laya-thai-decisions (2026-10-04)

[hagsmand1/laya-thai-decisions](https://huggingface.co/datasets/hagsmand1/laya-thai-decisions) (published 2026-10-03) is a
Thai typed-decision dataset built for laya by a third party: 41,776 training questions with soft targets (gold blended with
an LLM teacher) from MASSIVE, wisesight, wongnai, thaisum, iapp, eight synthetic workflows and synthetic NLI, and 17 frozen
eval suites. Its card lists its own weaknesses (H1-H5, M1-M8); licences are per source. `eval_decisions.py` runs the suites;
on the public checkpoint it reproduces the card's baseline (0.379 against 0.378 on MASSIVE th full, 0.113 against 0.118 on
negation), so the scoring matches. `results/decisions.json`.

| suite (cases) | public checkpoint | run 14 | run 16 (served) |
|---|---|---|---|
| MASSIVE th intent, 20 options (2,974) | 0.379 | 0.683 | 0.675 |
| MASSIVE th, scenarios absent from their train (796) | 0.436 | 0.752 | 0.741 |
| MASSIVE in ar / hi / ja / ko / ru / zh, mean (600) | 0.543 | 0.750 | 0.742 |
| MASSIVE th, unseen phrasings (1,092) | 0.577 | 0.654 | 0.655 |
| wisesight, unseen phrasings (1,052) | 0.461 | 0.639 | 0.661 |
| iapp reading comprehension, unseen phrasings (647) | 0.533 | 0.487 | 0.487 |
| unseen synthetic domains: insurance claims, school admin (300) | 0.560 | 0.582 | 0.553 |
| - of which choice / yes-no / 4-level score | 0.705 / 0.603 / 0.303 | 0.778 / 0.512 / 0.389 | 0.728 / 0.512 / 0.355 |
| probe: a claim and its negation get opposite answers (400) | 0.113 | 0.220 | 0.280 |
| probe: reversed 3-level scale gives the mirrored level (400) | 0.263 | 0.887 | 0.850 |
| probe: intent with 3 / 6 / 12 / 20 options (400) | 0.825 / 0.698 / 0.470 / 0.362 | 0.887 / 0.825 / 0.735 / 0.657 | 0.885 / 0.833 / 0.738 / 0.635 |

What it shows about our runs: on Thai text with questions like ours they are far ahead of the public checkpoint, also in
the other six languages, which nothing here trained. But on **questions they were not trained on** they are no better than
the public checkpoint: a yes/no question on an unseen domain is a coin flip (0.512), a 4-level scale is 0.36, reading
comprehension is below the baseline, and a negated yes/no question is answered as if it were not negated (0.35 on the
negated half). Fourteen runs on one fixed question set made a model for that question set. (`probe_noul_labels` says
nothing: 9 of 400 cases are true.)

## Run 18: training on laya-thai-decisions (2026-10-04)

`cs/label_cs18.py`, `cs/run18.sh`: the Thai part of their `train` split with its soft targets, 25,681 questions, plus 60,000
replayed run 16 items, a pass from run 16. Left out: thaisum (scraped news, publisher copyright per the card), the non-Thai
rows, and 100 rows whose text is in our own eval sets. `results/*18*`.

| their suites | run 16 (served) | run 18 |
|---|---|---|
| MASSIVE th intent, 20 options (2,974) | 0.675 | 0.733 |
| MASSIVE th, scenarios absent from their train (796) | 0.741 | 0.749 |
| MASSIVE in six other languages, mean (600) | 0.742 | 0.740 |
| MASSIVE th / wisesight / iapp, unseen phrasings | 0.655 / 0.661 / 0.487 | 0.849 / 0.737 / **0.910** |
| unseen synthetic domains: all, then choice / yes-no / 4-level score (300) | 0.553: 0.728 / 0.512 / 0.355 | **0.765**: 0.883 / 0.671 / 0.711 |
| probe: a claim and its negation get opposite answers | 0.280 | **0.938** |
| probe: reversed scale gives the mirrored level | 0.850 | 0.900 |
| probe: intent with 3 / 6 / 12 / 20 options | 0.885 / 0.833 / 0.738 / 0.635 | 0.958 / 0.897 / 0.855 / 0.757 |

| our sets | run 16 | run 18 |
|---|---|---|
| real posts, long: intent / department / urgency (false-other) | 0.614 / 0.722 / 0.811 (9.1%) | 0.606 / 0.711 / 0.806 (10.9%) |
| real posts, short: intent / department / urgency | 0.560 / 0.695 / 0.809 | 0.578 / 0.706 / 0.794 |
| reviews (300 hand labels): intent / department / urgency | 0.723 / 0.747 / 0.840 | 0.720 / 0.747 / 0.830 |
| cs9 held-out / public set (ECE) | 0.957 / 0.760 (0.158) | 0.954 / 0.761 (0.105) |
| Wisesight sentiment in our form: public / cs9 | 0.69 / 0.88 | 0.65 / 0.87 |
| reviews: stars agreement / frustration AUC | 0.861 / 0.910 | 0.872 / 0.916 |
| ambiguous set: all / sarcasm (100) / neutral facts (1,000) | 0.552 / 43 / 386 | 0.524 / 42 / 349 |
| held-out composed: sarcastic / sincere / indirect praise / plain negative | 0.98 / 0.99 / 0.98 / 1.00 | 0.98 / 0.99 / 0.98 / 1.00 |

Our triage does not move, which is the first requirement. On their suites the gains are large, but read them for what they
are: the "unseen phrasings" suites share sources and question kinds with their `train`, the negation probe is one question
("is it a question") turned around, and the unseen-domain suite keeps only what their own teacher agreed with (their H4).
`cs/new_questions_demo.py` asks six questions on twelve messages written for the check: run 16 gets 5/12, run 18 7/12. The
two it gains are churn and scam, kinds their `train` contains; on the kinds neither set has (a negated churn question in
Thai, a threat to go to the regulator, writing on behalf of someone else, a 4-level effort scale) both get 3 of 8. Twelve
cases prove little, but they do not show a model that follows any question. What run 18 buys is a wider set of known
question kinds and robustness to phrasing, option keys and scale direction - not zero-shot questions.

Not served: nothing on the desk asks those questions yet, the mood side loses a little (ambiguous set, Wisesight in our
form), and the licences of the new training data are per source (wongnai LGPL-3.0, teacher outputs of a commercial LLM).

## Decision 2.0 (vllm-sr), untrained, on our sets (2026-10-04/05)

[Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20) is a family of open decision models from the vLLM
Semantic Router team (published 2026-10-03): fine-tuned Qwen backbones from 0.6B to 27B with a decision head, Apache-2.0,
same request as ours (`system_one(state, questions)` with choice / noul / score, same answer fields), 8k-16k tokens of
input, loaded with `AutoModel(..., trust_remote_code=True)` on the transformers already in the training image. Their cards
report their own benchmarks only and say nothing about Thai. `cs/agents.py` gives the eval scripts one loader for both
kinds of model; `d2/eval_all.sh` ran Kai-0.6B, Sol-2B and Nox-4B beside the cascade on the A2, no training.
`results/d2*`.

| | run 16 (served) | Kai 0.6B | Sol 2B | Nox 4B |
|---|---|---|---|---|
| reviews (300 hand labels): intent / department / urgency | 0.723 / 0.747 / 0.840 | 0.537 / 0.597 / 0.550 | 0.167 / 0.537 / 0.553 | 0.317 / 0.710 / 0.530 |
| real posts, long: intent / department / urgency | 0.614 / 0.722 / 0.811 | 0.369 / 0.461 / 0.242 | 0.206 / 0.353 / 0.256 | 0.447 / 0.417 / 0.242 |
| real posts, short: intent / department / urgency | 0.560 / 0.695 / 0.809 | 0.370 / 0.443 / 0.244 | 0.209 / 0.372 / 0.265 | 0.381 / 0.423 / 0.244 |
| 1,000 reviews: stars agreement / frustration AUC | 0.851 / 0.909 | 0.672 / 0.845 | 0.614 / 0.703 | 0.656 / 0.796 |
| ambiguous set, all 3,300 / sarcasm (100) | 0.552 / 43 | 0.444 / 35 | 0.331 / 48 | 0.567 / 34 |
| our composed sarcastic / sincere | 0.98 / 0.99 | 0.34 / 0.77 | 0.68 / 0.29 | 0.51 / 0.81 |
| new questions (`cs/new_questions_demo.py`, 12) | 5 (run 18: 7) | 8 | 7 | 8 |
| laya-thai-decisions: unseen synthetic domains (300) | 0.553 (run 18: 0.765) | 0.733 | 0.652 | 0.719 |
| laya-thai-decisions: negation flips / reversed scale mirrors | 0.280 / 0.850 (run 18: 0.938 / 0.900) | 0.932 / 0.935 | 0.945 / 0.930 | 0.902 / 0.728 |
| laya-thai-decisions: MASSIVE th 20 options (100) / wisesight phrasings | 0.930 / 0.661 | 0.730 / 0.617 | 0.530 / 0.570 | 0.800 / 0.603 |
| one request with intent + department + urgency, on the A2 | 0.03 s | 1.4 s | 2.0 s | 4.8 s |
| one short yes/no or score question, on the A2 | 25 ms | 39 ms | 224 ms | 305 ms |
| GPU memory | 3 GB | 2 GB | - | - |

- **On our questions the trained laya is far ahead of all three**, as a model trained on those questions should be. On
  urgency they do nothing: the probabilities are flat, the rounded expectation is the middle level 292 times of 300 (0.55 is
  the share of that level), and the most probable level is right 0.35-0.40 of the time (`d2/urgency_check.py`).
- **On questions nobody trained on they are ahead of run 16** and level with run 18, which was trained on that dataset:
  negation and scale direction work out of the box, an unseen domain is 0.65-0.73 against 0.55, the twelve new questions 7-8
  against 5. That is what a foundation decision model is for, and it holds in Thai.
- **Bigger is not better in Thai here.** Sol-2B is the worst of the three on almost every Thai set (intent 0.17-0.21) and
  Nox-4B does not beat Kai-0.6B overall. The sizes have different backbones (Qwen3, Qwen3.5); their published ranking is on
  their own, mostly English, benchmarks.
- **Speed on the A2 is the opposite of the card**: 5 ms on their GPU, 39 ms here for one short question and 1.4 s once the
  45-option intent question is in the request, because every question is a separate pass over a 600M-parameter decoder.

Not a replacement for laya on the desk, and not usable as a labelling teacher for our questions (worse than the model it
would teach). The use that fits: Kai-0.6B next to laya for a yes/no question the desk has not asked before, at 40 ms and
2 GB, until there are examples to train that question into laya.

## Clef-Flash (Cloudflare, 9B), untrained, 4-bit on the A2 (2026-10-06)

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (2026-10-01, Apache-2.0) is a decision model on
Qwen3.5-9B with a "joint schema head" that scores every option of every question in one forward pass; same request and
answer fields as ours. Its card reports English benchmarks and says nothing about Thai. The 27B Clef is 55 GB and out of
reach here. The 9B is 19 GB in BF16, so it runs 4-bit (bitsandbytes NF4, `d2/Dockerfile.clef`), which the card does not
test: 8.7 GB peak, beside the cascade. `cs/agents.py` loads the repo's own `joint_schema_model.py` minus its image / video
processor (torchvision is not in the image; text records never touch it). Same `d2/eval_all.sh` as Decision 2.0.
`results/d2_clef-flash_*`, `results/clef.log`.

| | run 16 (served) | Decision 2.0 Kai 0.6B | Clef-Flash 9B, 4-bit |
|---|---|---|---|
| new questions (`cs/new_questions_demo.py`, 12) / the 8 nobody trained | 5 / 3 (run 18: 7 / 3) | 8 / 4 | **10 / 7** |
| laya-thai-decisions: unseen synthetic domains (300) | 0.553 (run 18: 0.765) | 0.733 | **0.777** |
| laya-thai-decisions: negation flips / reversed scale mirrors | 0.280 / 0.850 | 0.932 / 0.935 | 0.868 / 0.880 |
| laya-thai-decisions: MASSIVE th 20 options (100) / wisesight phrasings | 0.930 / 0.661 | 0.730 / 0.617 | **0.960** / 0.657 |
| ambiguous set, all 3,300 / neutral called neutral (of 1,100) | 0.552 / 402 | 0.444 / 464 | **0.612 / 691** |
| ambiguous set: sarcasm (100) / our composed sarcastic | 43 / 0.98 | 35 / 0.34 | 21 / 0.41 |
| 1,000 reviews: stars agreement (opposite polarity) / frustration AUC | 0.851 (0.106) / 0.909 | 0.672 (0.093) / 0.845 | 0.647 (0.068) / 0.893 |
| reviews (300 hand labels): intent / department / urgency | **0.723 / 0.747 / 0.840** | 0.537 / 0.597 / 0.550 | 0.403 / 0.667 / 0.767 |
| real posts, long: intent / department / urgency | **0.614 / 0.722 / 0.811** | 0.369 / 0.461 / 0.242 | 0.400 / 0.547 / 0.767 |
| real posts, short: intent / department / urgency | **0.560 / 0.695 / 0.809** | 0.370 / 0.443 / 0.244 | 0.387 / 0.541 / 0.740 |
| `other` caught: reviews (142) / long posts (95) | 131 / 74 | 97 / - | 34 / 25 |
| one request with intent + department + urgency, on the A2 | 0.03 s | 1.4 s | 3 s |
| one short question, on the A2 | 25 ms | 39 ms | 650 ms |
| GPU memory | 3 GB | 2 GB | 8.7 GB |

- **The best model here on questions it was never given**: 7 of the 8 untrained new questions (a 4-level "customer effort"
  scale both right, a threat to go to the regulator, writing for someone else), 0.78 on the unseen domains, and it reads a
  neutral sentence as neutral. Unlike Decision 2.0 it also does our urgency untrained (0.74-0.77; asked alone 0.68) and its
  frustration ranks reviews almost as well as the trained model (AUC 0.893 against 0.909).
- **On the desk's own questions laya is still ahead**, by 15-20 points on department and by more on intent. Most of the
  intent gap is one decision: Clef-Flash almost never answers `other` (34 of 142 out-of-scope reviews caught, laya 131). On
  the in-scope reviews alone the two are level (0.55 and 0.54); on in-scope long posts laya leads 0.55 to 0.45.
- Sarcasm of the "praise, then the fault" kind is not there untrained (0.41 on our composed set; star agreement 0.65 with
  42% of 2-star reviews called neutral).
- **Too slow and too large to serve here**: 3 s for the desk's three questions and 0.65 s for one, 4-bit, on a card it
  shares; the card's 39 ms is an H200 at full precision. 4-bit also means these numbers are a lower bound on the model.

What it is good for on this hardware is offline work: a labelling teacher for a **new** question (churn threat, regulator
threat, third party, effort) over our real texts, whose answers are then trained into laya - the recipe of runs 13-14 with
a teacher that, for new questions, is better than anything tried before. For the questions laya already has it would be
teaching a better student.

## Run 19: three questions laya had never been asked (2026-10-06/07)

The desk may want questions the fixed set does not have. Four were tried: a churn threat, a threat to go outside (regulator,
court, formal complaint), writing on behalf of someone else, and a 4-level "how much has the customer already had to chase
this" (`cs_questions.EXTRA`). The recipe of runs 13-14: a larger model labels our real texts, laya is trained on the labels,
a hand-labelled set that stayed out of training is the test. The teacher was checked **before** anything was trained:
307 texts labelled by hand (`cs/make_new_labels.py`, two samples: 127 drawn by the first teacher's answers, 180 drawn by
cue words so that a "yes" is not rare), `cs/eval_new.py`.

| teacher on the 307 (yes said / right / found of hand yes) | churn threat (49 yes) | outside threat (7) | third party (20) | contact effort (57 with contact) |
|---|---|---|---|---|
| Clef-Flash 4-bit, first wording (127 texts only) | 60 / 5 / 5 of 5 | 30 / 1 / 1 of 1 | 7 / 2 / 2 of 6 | 77 / 14 / 14 of 14 |
| Qwen3-4B, written rules | 33 / 19 / 19 of 40 | 12 / 6 / 6 of 8 | 19 / 11 / 11 of 20 | 23 / 20 / 20 of 57 |
| Qwen3-8B, written rules, first definitions | 30 / 23 / 23 of 40 | 10 / 5 / 5 of 8 | 18 / 10 / 10 of 20 | 37 / 34 / 34 of 57 |
| **Qwen3-8B, written rules, final definitions** | **51 / 39 / 39 of 49** | 6 / 5 / 5 of 7 | 17 / 10 / 10 of 20 | **62 / 46 / 46 of 57** |

Clef-Flash, the best model on the twelve made-up questions a day earlier, reads an angry review as a threat to leave
(5 right of 60) - its negatives in that demo were too clean to show it. A generative model that reads the rules does far
better, and the 8B is needed (the 4B misses "ต้องการย้ายค่ายจาก dtac ไปทรู"). Part of the first gap was the definitions:
"leaving" depended on which company is being asked, which the text does not say, so it became "talks about leaving the
provider they use now, including asking for port-out deals". The rules were tuned on these same 307 texts, so the final
row is a little optimistic. Third party stayed at 10 of 17 and was dropped. The eval set is small on the outside threat
(7 yes).

Items (`cs/label_cs19.py`): Qwen3-8B over 27,716 pool texts (`cs/label_new_gen.py`, 2 h 8 min beside the cascade), the
307 hand-checked texts left out; per question every "yes" (churn 2,113, outside threat 51 x6, contact 2,414), every "no"
with a cue word and random "no" to 3:1; 19,536 new items + 60,000 replayed run 16 items, a pass from run 16.
`results/new19.json`, `results/*19*`.

| on the 307 hand-labelled texts (said / right / found) | teacher (Qwen3-8B) | run 16 | **run 19** |
|---|---|---|---|
| churn threat | 51 / 39 / 39 of 49 | 19 / 6 / 6 of 49 | **51 / 39 / 39 of 49** |
| outside threat | 6 / 5 / 5 of 7 | 66 / 5 / 5 of 7 | 32 / 4 / 4 of 7 |
| contact effort: exact level / within one / any contact right | 266 / 302 / 280 of 307 | 76 / 126 / 115 | 249 / 296 / 261 |
| contact effort: says contact / right / found of 57 | 62 / 46 / 46 | 235 / 50 / 50 | 79 / 45 / 45 |
| twelve made-up new questions (other wordings) | - | 5 | 6 |

| our sets | run 16 | run 19 |
|---|---|---|
| real posts, long: intent / department / urgency | 0.614 / 0.722 / 0.811 | 0.603 / 0.711 / 0.814 |
| real posts, short | 0.560 / 0.695 / 0.809 | 0.562 / 0.707 / 0.804 |
| reviews (300 hand labels) | 0.723 / 0.747 / 0.840 | 0.737 / 0.743 / 0.817 |
| stars agreement / frustration AUC / ambiguous set | 0.861 / 0.910 / 0.552 | 0.872 / 0.916 / 0.553 |
| cs9 held-out / public set | 0.957 / 0.760 | 0.951 / 0.757 |

The churn question is learned to the teacher's level exactly (the same 39 of 51 and 39 of 49), contact effort nearly
(81% exact level against the teacher's 87%), and nothing the desk already has moved beyond noise. The outside threat did
not take: 51 positive examples, even repeated, give a model that says yes 32 times for 4 hits; it needs more examples or a
different source. A churn question answered 0.03 s after the message, from laya, is what this was for.

**Served since 2026-10-07: `:8011` runs run 19** (roll back by mounting `out/laya-th-run16`). The test desk (`ui/`) now shows
the two new answers on the slip: "พูดถึงการเลิกใช้ / ย้ายค่าย" and "ตามเรื่องนี้มาแล้ว" (4 levels); the outside threat is not
shown. Spot check through the desk: "เน็ตหลุดทุกวัน โทรแจ้งไปสามรอบแล้ว ... จะย้ายค่ายแล้วนะ" -> churn 90%, contacted before;
"อยากสมัครบัตรเครดิต" -> churn 11%, no contact.

## Reflex (gist.rs) and the MegaWiz "decision ladder" (2026-10-07)

[Reflex](https://reflex.gist.rs) is a modelless decision engine (Rust, MIT): a state is scored against a corpus of
markdown documents per domain by compression distance and cosine similarity, answers carry a calibrated confidence and
**abstain is a first-class output**. Same question kinds as laya (`noul` / `choice` / `score`; the wire is
`{"state", "questions": [{"id", "kind", "prompt"}]}` on `POST 127.0.0.1:7331/decide`). Installed v0.2.4 on the box and
asked it a Thai ticket: `outcome: null, confidence 0.00009`. That is by design and documented in the engine's own
`.docs/02_protocols/thai_posture.md`: tokens are split on ASCII whitespace and trimmed to ASCII alphanumerics, so a Thai
sentence embeds to the zero vector and the distance gate abstains; "the engine's Thai capability is genuinely absent, not
collapsed" (their wording), `thai_wisesight` 0.12 against chance 0.25. Nothing to evaluate on our sets; the binary is left in
`~/reflex` on the box, server stopped.

The MegaWiz post ([Ultra Instinct ของบันไดตัดสินใจ](https://asgard.megawiz.co.th/blog/ultra-instinct-decision-ladder-th),
2026-10-06) is the useful part: they built their own Thai ladder (`MegaWiz-Dev-Team/reflex-study`, AGPL-3.0, character
n-gram tokeniser) - rung 0 an Instinct-style NBSVM classifier on CPU that answers when a conformal gate is confident, then
an encoder, then an LLM read by first-token log-probabilities (~0.3 s), with "abstain, hand the set of still-possible
labels to the teacher" at the top. Their numbers are on a 168-row synthetic 8-class task with a 64-row test set (±10
points): NBSVM 87.5% against a bag-of-words 85.9%, not separable; distillation looked like a win until a second code
review found leakage in the cross-validation, after which it called the LLM almost twice as often. What transfers to us:
abstain as a designed output (our cascade's 0.7 threshold is that, unmeasured so far), a cheap rung in front of laya only
where it is provably better, and "a result that is better than expected is a signal to read the code again".

## A bigger teacher: Qwen3.8-27B at 2 bits (2026-10-10)

`ConwayResearch/Underdog-Saluki-27B-1.0` is Qwen3.8-27B (Aug 2026, two generations after the Qwen3-8B that taught run 19)
cut to 7.9 GB (IQ2 GGUF), tuned to keep tool calling, Apache-2.0. It runs on llama.cpp (`cs/saluki_up.sh`, :8013) beside
the cascade: 12.7 GB on the card together. The same 307-text check as before (`label_new_gen.py --eval --url
http://localhost:8013`, same rules), scored with `eval_new.py`:

| teacher on the 307 (said / right / found of hand yes) | churn threat (49) | outside threat (7) | third party (20) | contact effort: exact level / says contact right of said / found of 57 |
|---|---|---|---|---|
| Qwen3-8B FP8 (run 19's teacher) | 51 / 39 / 39 | 6 / 5 / 5 | 17 / 10 / 10 | 266 of 307 / 46 of 62 / 46 |
| **Qwen3.8-27B 2-bit (Saluki)** | **42 / 41 / 41** | 9 / 7 / **7** | 30 / 19 / **19** | **288 of 307 / 51 of 57 / 51** |
| time for the 307 | 1.7 min | | | 24.7 min |

Better on every question; what it misses on churn are mostly the texts the hand labels themselves found borderline
(port-in requests, cancelling a line in a parent's name), and it over-says third party on news and adverts (11 extra),
which a rule can trim. The price is speed: about 1.8 tokens/s generation on the A2, 10x slower than the 8B, so the whole
27,716-text pool would take ~37 h; labelling only the texts with a cue word (~4,500) is ~6 h. The two questions dropped
after run 19 (outside threat, third party) are teachable with this teacher. `data_domain/new_teacher_saluki_eval.jsonl`.

Qwen3.5-9B-FP8 (the 8B's direct successor) does not fit: 14 GB with its vision tower and 248k vocabulary, out of memory
beside the cascade; its 4-bit build (`RedHatAI/Qwen3.5-9B-quantized.w4a16`) is still 11 GB for the same reason and also runs out of
memory next to the cascade. Either would need the card to itself (cascade stopped), untested.

## Abstain, measured (2026-10-07)

`cs/eval_abstain.py` and `cs/eval_abstain2.py` on the 1,374 hand-labelled real rows (360 long posts, 714 short versions,
300 reviews), run 19, with the teacher (OpenThai-SystemOne, :8010) asked the same questions. `results/abstain19*.json`.

| intent, all rows | answers itself | right when it does | right on the handed-over part | teacher on that part | cascade | hand to a person |
|---|---|---|---|---|---|---|
| threshold 0.7 (served) | 96% | 62.3% | 37.7% | 24.6% | 60.6% | 64.0% |
| threshold 0.8 | 94% | 62.8% | 34.6% | 24.4% | 60.6% | 64.9% |
| threshold 0.9 | 0% | - | 61.2% | 32.5% | 32.5% | - |
| student alone / teacher alone | | 61.2% | | 32.5% | | |

- **The served threshold hands over almost nothing**: the largest probability sits between 0.8 and 0.9 on nearly every
  answer (label smoothing and the temperature put it there), so 0.7 passes 96% and 0.9 passes 0%. The cascade is laya alone.
- **The teacher fallback costs accuracy**: on the part laya is unsure about, the teacher is right 24-33%, laya itself
  35-50%. True for all three questions and all three sets. Turn it off, or hand over to a person instead.
- **Confidence is weakly informative** (AUROC right-vs-wrong: intent 0.62-0.64, department 0.71-0.72, urgency 0.65; the
  gap between the top two probabilities is the best of the signals, barely). Handing the least confident 20% to a person
  lifts what laya keeps from 61% to 67% on intent, 72% to 79% on department, 81% to 85% on urgency; the handed part is right
  37-64% of the time, so a person does earn their share. Abstain is worth having as a workload dial, not as a safety net.

## Run table

Every run on every set it was measured on, in percent, from `results/` (`cs/runs_table.py --write`). Long = 360 hand-labelled Pantip posts, short = their 714 chat versions, reviews = 300 hand-labelled Google Play reviews, stars = 1-2 stars called negative or 4-5 positive (3,000 reviews, from run 15), ambiguous = 3,300 sentences of Tippawan/thai-ambiguous-sentiment, cs9 = 11,720 held-out decisions of the synthetic call-center set, public = the 2,455 public-dataset decisions. Differences under about 3 points on the 360-post set are noise.

| run | what changed | long posts: intent / dept / urgency | short: intent / dept / urgency | reviews: intent / dept / urgency | stars | ambiguous | cs9 | public |
|---|---|---|---|---|---|---|---|---|
| 3 | bigger distillation set + human labels, option budget 768 | - | - | - | - | - | - | 77.1 |
| 4 | call-center distillation | - | - | - | - | - | - | 78.3 |
| 5 | calibration, more score items, an `other` intent | - | - | - | - | - | - | 78.2 |
| 6 | telecom / banking / insurance / e-commerce | - | - | - | - | - | - | 78.5 |
| 7 | in-register out-of-scope texts | - | - | - | - | - | - | 78.2 |
| 8 | first real Thai in-domain text (Pantip), served 09-28 to 10-02 | 57 / 65 / 75 | 52 / 59 / 60 | - | 63 | 49 | 96.8 | 78.0 |
| 9 | 1,000 more hand-labelled Pantip rows | - | - | - | - | - | 95.4 | 77.6 |
| 10 | short second pass on the real rows | 55 / 68 / 80 | 50 / 65 / 72 | - | - | - | 96.3 | 77.3 |
| 11 | short chat versions, human frustration labels, hard examples | 58 / 70 / 80 | 55 / 69 / 79 | 56 / 62 / 47 | - | - | 96.5 | 77.3 |
| 12 | Google Play reviews, teacher + student agreement labels | 56 / 69 / 80 | 55 / 69 / 76 | 63 / 62 / 25 | - | - | 96.2 | 76.9 |
| 13 | reviews labelled by Qwen3-8B | 56 / 68 / 82 | 55 / 70 / 80 | 74 / 74 / 81 | - | - | 95.5 | 75.7 |
| 14 | Qwen labels on forum posts where the student agrees, served 10-02 to 10-03 | 56 / 72 / 81 | 56 / 71 / 79 | 74 / 74 / 82 | 72 | 49 | 95.7 | 76.3 |
| 15 | composed sarcasm + star-labelled mood; neutral collapsed, not served | 59 / 73 / 81 | 55 / 71 / 80 | 73 / 75 / 83 | 88 | 51 | 94.9 | 74.4 |
| 16 | 15 + Wisesight neutral / question, served 10-03 to 10-07 | 61 / 72 / 81 | 56 / 69 / 81 | 72 / 75 / 84 | 86 | 55 | 95.7 | 76.0 |
| 17 | 16 with neutral x2; long-post triage down, not served | 58 / 70 / 79 | 56 / 70 / 79 | 73 / 75 / 83 | 84 | 55 | 95.5 | 77.0 |
| 18 | 16 + hagsmand1/laya-thai-decisions Thai split, not served | 61 / 71 / 81 | 58 / 71 / 79 | 72 / 75 / 83 | 87 | 52 | 95.4 | 76.1 |
| 19 | 16 + churn threat and contact effort from Qwen3-8B rules, served since 10-07 | 60 / 71 / 81 | 56 / 71 / 80 | 74 / 74 / 82 | 87 | 55 | 95.1 | 75.7 |

## Syncing with upstream laya (2026-10-07)

The fork point is 2026-09-25 (0.3.20); upstream `NandhaKishorM/laya` is 1,050 commits further (0.3.28: an official trainer,
order-invariant option layout, calibration / abstention fixes, verify tools, integrations). Nothing outside `thai/` was
changed here, so `upstream/main` merges into branch `sync-upstream` without a conflict. Before the merge touches the
served checkpoints: `cs/parity_check.py` / `cs/parity.sh` run run 19 under both trees on 200 texts x 7 questions -
**1,400 answers identical, largest probability gap 0.0012**; `eval_play` and `eval_real` under 0.3.28 reproduce the run 19
numbers to the digit; every laya symbol our scripts import still exists. A side finding: the training box's copy of the
package (`~/laya/laya`, synced by scp) has been 0.3.5 all along, so runs 3-19 were trained and evaluated on 0.3.5 code;
the parity result covers that gap too (0.3.5 vs 0.3.28).

**Merged 2026-10-07**: `thai` now carries upstream 0.3.28, and `~/laya` on the box is a git clone of it (`git init` in place,
`git checkout -f thai`; the untracked model, data and feedback folders stayed where they were). From now on the box is
synced with `git pull`, not scp. The serving image (`laya-cascade:run8`) is unchanged; the next training run uses 0.3.28.

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

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

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

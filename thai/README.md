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
| `cascade_server.py`, `Dockerfile.cascade`, `docker-compose.cascade.yml`, `smoke_cascade.py`, `bench_cascade.py` | **the cascade as a service**: same `/v1/systemone` contract as the teacher, student on GPU 1 at `:8011`, teacher at `:8010`; smoke test with a 60-option question and 8 concurrent callers |
| `rewrite_colloquial.py`, `check_rewrites.py`, `run_rewrite.sh` | run 4 data: rewrite the Thai Bitext customer-support set into spoken/chat Thai with a local LLM (vLLM), then let the teacher check that each rewrite still carries its intent |
| `label_cc.py`, `run4.sh`, `cascade4.sh` | run 4: the call-center question set labelled by two teacher instances, items, grouped eval split, train from run 3, eval |
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

Deployed on the dev box next to the teacher (2026-09-23 with run 3, **run 4 since 2026-09-24 18:40**): `docker compose -f thai/docker-compose.cascade.yml up -d --build`
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

Reading: run 5 is the checkpoint to serve for call-center triage (same in-domain accuracy as run 4, out-of-scope detection, better
score agreement, temperatures near 1) and equal to run 4 elsewhere. Still unmeasured on real tickets.

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

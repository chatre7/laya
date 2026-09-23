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
`results/run3.log`, `results/distill3_manifest.json`. Checkpoint `thai/out/laya-th-run3` on the dev box, not published.

**Reading.** Overall accuracy 0.719 -> 0.772 (teacher 0.814) and Brier 0.400 -> 0.314, with calibration still good.
The gain is in-domain: massive_th 0.54 -> 0.86 (the 768-token option budget plus wide-intent questions did what
they were meant to), prachathai +10 points from seeing its human labels again. The held-out sets did not improve:
wisesight 0.587 -> 0.557 (still at the teacher's 0.547), sib200 0.745 -> 0.701, so the extra 2.5x teacher data and the
human labels bought no new generalisation, and sib200 hints at mild over-fitting to the training corpora. Tickets:
the same login ticket is still routed to billing (p=0.70); everything else is right and frustration MAE is the best so far.
Cascade numbers (next section) are from run 2 and have not been re-measured with this checkpoint; with the student
alone at 0.772 the gate should need fewer teacher calls for the same accuracy.

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

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

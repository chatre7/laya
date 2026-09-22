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
| `results/` | json summaries and the training log of every run |

```bash
bash thai/run.sh build
bash thai/run.sh prep 1500                                   # human-labelled set (needs thai/ots)
bash thai/run.sh distill --per-source 3000                   # teacher-labelled set, background
bash thai/run.sh train-distill --epochs 2                    # background
bash thai/run.sh eval-distill /work/thai/out/laya-th-distill distill
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

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

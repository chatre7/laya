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

## Run 2: distillation from OpenThai-SystemOne (in progress)

Instead of human labels, `distill_from_ots.py` sends Thai texts from six public corpora with 2–4
questions each from a bank of ~20 question types (sentiment, intent, department, urgency, frustration,
topic, rating, NLI, several yes/no checks; paraphrased instructions, option subsets of varying size) to the
OpenThai-SystemOne server and keeps its probabilities as soft targets. The teacher answers at
~100 ms per record, so the set costs GPU time, not annotation. The student can at best match the
teacher, but at ~40 ms and 322M parameters. It has no abstain slot, so the teacher's abstain mass is
dropped and the remaining probabilities renormalised.

Results: _pending_ (see `results/distill*.json` when done).

## Known limits of laya for our use

- No abstain output (OpenThai's browser-agent demo depends on it).
- All options of a question share a 256-token budget (`head_max_len`), so 30+ options (web page
  elements) degrade to a few tokens per option. Fixing this is an architecture change, not a config.
- `laya-multilingual`'s encoder (mmBERT-base) had no Thai continued pre-training.

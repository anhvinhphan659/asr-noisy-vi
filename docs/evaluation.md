# Evaluation (`evaluation`)

`evaluation` turns what the models heard into the numbers of the report: WER and CER per
condition, tables comparing systems, and bootstrap intervals that say whether a difference
is real or test-set luck. It does not run any model.

## 1. Where it fits

```
 test plan (dataprep) ──▶ model runs on GPU ──▶ predictions_<model>_<variant>.csv
                                                         │
 clean manifest (references) ────────────────────────────┤
                                                         ▼
                                       evaluation: score → summarize / wer_table
                                                         → bootstrap_ci / compare
```

It is used twice: after the zero-shot runs (baseline), and after fine-tuning (zero-shot vs
fine-tuned on clean vs fine-tuned with noise).

Model runs only **write raw outputs**. Scoring happens in one place, with one normalization,
for every system. To change how text is normalized, scoring is rerun in seconds on CPU;
models do not need to run again.

| Module | Purpose |
|---|---|
| `metrics.py` | Substitutions, deletions and insertions between two texts (words, characters) |
| `predictions.py` | Write model outputs in the prediction format, resuming interrupted runs |
| `scoring.py` | Load predictions, join them with the plan and references, count errors, aggregate |
| `bootstrap.py` | Confidence intervals for one system, paired comparison of two systems |

## 2. Prediction files

Each model run writes one CSV, one row per (utterance, condition) of the test plan:

| Column | Example |
|---|---|
| `utt_id` | `fleurs_10056093347220194289` |
| `condition` | `clean`, `traffic_5dB`, `babble_0dB` (from the test plan) |
| `model` | `qwen3-asr-0.6b`, `phowhisper-small` |
| `variant` | `zero-shot`, `ft-clean`, `ft-noisy` |
| `hyp` | Raw model output, **not normalized** |

`PredictionWriter` (`predictions.py`) writes this format. It appends and flushes after every
batch, and on a new run skips the rows already written, so a Kaggle session that stops
halfway is resumed by running the notebook again:

```python
from dataprep import iter_eval_batches, load_eval_set
from evaluation import PredictionWriter

eval_set = load_eval_set(MANIFEST_DIR / "test_plan.csv", MANIFEST_DIR / "clean.csv")
with PredictionWriter(f"predictions_{MODEL}_{VARIANT}.csv", MODEL, VARIANT) as writer:
    for rows, audios in iter_eval_batches(writer.pending(eval_set), DATA_ROOT):
        writer.add(rows, transcribe(audios))        # transcribe: model-specific
```

It refuses a file that already holds another system's predictions, and writes each
hypothesis on one line. After `pending`, it prints a progress line about once a minute (rows
written against rows pending, rate, time left; see `dataprep.set_progress_interval`).

`load_predictions(paths)` stacks the files and rejects missing columns or duplicated rows.
`missing_predictions(predictions, plan)` lists the plan rows a system did not cover, so an
interrupted run is noticed before scoring.

## 3. Metrics

**WER** (word error rate) = (S + D + I) / N, where N is the number of reference words, and
S, D, I are the substitutions, deletions and insertions of the minimum edit alignment.

```
Reference:  hôm nay trời đẹp quá     (N = 5)
Hypothesis: hôm nay chời đẹp         S = 1 (trời → chời), D = 1 (quá)
WER = 2 / 5 = 40%
```

**CER** (character error rate) is the same over characters, spaces included (as in `jiwer`).
It is useful for Vietnamese: a wrong tone ("trời" / "trơi") is one wrong word but only one
wrong character out of four.

Rules:

- Reference and hypothesis both go through `dataprep.normalize_text` (lowercase, numbers
  written as words, no punctuation, unified tone placement and number-word variants; see
  `dataprep.md`). A model that writes "2011" and a reference that says "hai nghìn không
  trăm mười một" therefore match.
- Rates are **corpus-level**: errors and reference lengths are summed over a group, then
  divided. They are not averages of per-utterance rates, which would overweight short
  utterances.
- WER can exceed 100% when a model inserts many words (e.g. transcribing babble voices).
- Rates are in percent.

Edit distances are computed with `rapidfuzz` (C implementation), fast enough for tens of
thousands of utterances per system.

## 4. Scoring and tables

```python
from evaluation import load_predictions, missing_predictions, score, summarize, wer_table

plan = pd.read_csv(MANIFEST_DIR / "test_plan.csv")
manifest = pd.read_csv(MANIFEST_DIR / "clean.csv")
predictions = load_predictions(Path("predictions").glob("*.csv"))
assert missing_predictions(predictions, plan).empty

scored = score(predictions, plan, manifest)   # one row per prediction, with error counts
summary = summarize(scored)                   # model × variant × dataset × noise × SNR
table = wer_table(scored)                     # report table
```

`score` adds `dataset`, `noise_type`, `snr_db` (from the plan), `ref` and `hyp_norm`, and the
counts `w_sub`, `w_del`, `w_ins`, `w_n_ref` (words) and `c_*` (characters).

`summarize(scored, by=...)` gives per group: `utterances`, `words`, `wer`, `cer`, and
`sub_pct`, `del_pct`, `ins_pct` (WER split into its three error kinds). The split helps the
analysis: babble at low SNR typically raises insertions (the model transcribes the
background voices), traffic raises deletions.

`wer_table` is the main report table: one row per dataset × noise type × SNR, one column per
model / variant.

| dataset | noise_type | snr_db | qwen zero-shot | qwen ft-clean | qwen ft-noisy | pho zero-shot | … |
|---|---|---|---|---|---|---|---|
| fleurs | clean | – | | | | | |
| fleurs | traffic | 20 | | | | | |
| … | | | | | | | |

The same table drawn as **WER versus SNR**, one line per system, answers the research
questions directly: the slope shows how fast a system degrades, the gap between `ft-clean`
and `ft-noisy` at low SNR shows what noise-aware fine-tuning adds.

Other cuts use `by`, e.g. `summarize(scored[scored.dataset == "vietmed"],
by=["model", "variant", "noise_type"])`, or merge `rec_condition` or `has_digit` from the
manifest first.

## 5. Bootstrap

A WER is measured on one test set. With a different sample of sentences it would be a bit
different. The bootstrap estimates by how much:

1. draw, with replacement, as many utterances as the group has;
2. recompute the corpus-level WER on that resample;
3. repeat 1000 times; the 2.5th and 97.5th percentiles give a 95% interval.

`bootstrap_ci(scored, by=...)` gives `wer`, `ci_low`, `ci_high` per group.

`compare(scored, system_a, system_b)` is a **paired** bootstrap of `WER(b) − WER(a)` per
condition: both systems are scored on the same utterances and resampled with the same
indices, which removes the variation that comes from sentence difficulty.

```python
from evaluation import compare

compare(scored, ("qwen3-asr-0.6b", "ft-clean"), ("qwen3-asr-0.6b", "ft-noisy"))
```

| Column | Meaning |
|---|---|
| `wer_a`, `wer_b`, `diff` | WER of each system and `b − a`; negative means b is better |
| `ci_low`, `ci_high` | 95% interval of the difference |
| `significant` | The interval excludes 0 |

If the interval contains 0, the report should not claim that one system is better under that
condition.

## 6. Comparisons for the report

| Question | `system_a` | `system_b` |
|---|---|---|
| Does fine-tuning help? | `(m, "zero-shot")` | `(m, "ft-clean")` |
| Does noise in training help further? (main question) | `(m, "ft-clean")` | `(m, "ft-noisy")` |
| Does noise training hurt clean speech? | `(m, "ft-clean")` | `(m, "ft-noisy")`, on the `clean` rows |
| Which model is more robust? | `("qwen3-asr-0.6b", v)` | `("phowhisper-small", v)` |

For a fair comparison, all systems must share the test plan, the training schedule, the
training budget and the checkpoint selection rule (see `dataprep.md`).

## 7. For the report

- WER and CER are corpus-level, in percent, after `normalize_text` on both sides.
- Numbers are written as words on both sides; codes glued to letters (`covid19`) are kept.
  Readings the normalization cannot match (years read digit by digit, times, dates) remain:
  report FLEURS results with and without `has_digit` utterances (merge the column from the
  clean manifest on `utt_id`).
- Intervals are 95% percentile bootstrap intervals over utterances, 1000 resamples, seed 42.
- Conditions come from the test plan: the same noise segment is used at every SNR, so
  differences across SNR levels come from the SNR alone.

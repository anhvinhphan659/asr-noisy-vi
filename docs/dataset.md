# The shared dataset `asr-noisy-vi`

This page is for anyone who writes a model notebook (zero-shot run, fine-tuning, test,
demo). It describes what the private Kaggle Dataset `asr-noisy-vi` contains and how to load
it for each task. How the data was built is in `dataprep.md`; how outputs are scored is in
`evaluation.md`.

The dataset is built once by notebook `01_prepare_dataset`. Model notebooks never build or
change data: they attach the dataset and read it.

## 1. Setup in a Kaggle notebook

Attach two inputs (*Add Input → Datasets → Shared with you / Your Datasets*):

| Input | Content |
|---|---|
| `asr-noisy-vi` | The data (this page) |
| `asr-noisy-vi-code` | The packages `core`, `dataprep`, `evaluation` |

First cell: install the code (inputs are read-only, so it is copied to `/tmp`).

```python
import os, shutil, subprocess, sys
from pathlib import Path

def find_code(root=Path("/kaggle/input")):
    for depth in range(1, 6):
        for hit in sorted(root.glob("/".join(["*"] * depth) + "/dataprep/__init__.py")):
            if (hit.parents[1] / "pyproject.toml").is_file():
                return hit.parents[1]
    raise FileNotFoundError("attach the Kaggle Dataset 'asr-noisy-vi-code'")

code_dir = Path("/tmp/asr-noisy-vi")
shutil.copytree(find_code(), code_dir, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{code_dir}[notebooks]"],
               check=True)
sys.path.insert(0, str(code_dir))     # importable in this kernel without a restart
```

Second cell: find and check the data.

```python
from dataprep import describe, find_data_root, require_data

DATA_ROOT = find_data_root()          # the folder holding DATASET.json, wherever Kaggle mounts it
info = require_data(DATA_ROOT)        # stops with a clear message if anything is missing
print(describe(info))                 # log this line with your results
MANIFEST_DIR = DATA_ROOT / "manifests"
```

Do not reinstall NumPy or PyTorch. On a local machine, `find_data_root()` also finds
`../asr-data` or `asr-data`, or the folder in `$ASR_DATA_ROOT`.

## 2. What is inside

```
asr-data/
├── DATASET.json      version, code commit, build parameters, row count of every manifest
├── LICENSES.md       sources and licenses (private use only: VIVOS and some FSD50K clips are non-commercial)
├── manifests/        the tables below (CSV); every path in them is relative to asr-data/
├── clean/<dataset>/<split>/<utt_id>.wav                  all utterances, clean
├── noisy/<dataset>/<split>/<condition>/<utt_id>.wav      test and dev only, mixed in advance
└── noise/
    ├── traffic/fsd50k_<id>.wav                           traffic clips (FSD50K)
    └── voices/musan_speech-librivox-<id>.wav             voices for babble (MUSAN)
```

All audio is **16 kHz, mono, 16-bit WAV**. `load_audio` returns it as a `float32` array in
[-1, 1]. Transcripts are already normalized (lowercase, no punctuation, numbers as words).

Version v1, built 2026-11-01:

| Folder | Files | Size |
|---|---|---|
| `clean/` | 25,740 | 5.1 GB |
| `noisy/` | 24,113 | 5.6 GB |
| `noise/` | 1,655 | 3.0 GB |
| total | 51,520 | 13.8 GB |

### Speech

| Dataset | Audio | train | dev | test (used) |
|---|---|---|---|---|
| FLEURS `vi_vn` | read speech, 16 kHz | 2,990 utt, 9.0 h | 359 utt, 1.2 h | 850 utt, 2.9 h |
| VIVOS | read speech, quiet room, 16 kHz | 10,626 utt, 13.8 h | 1,034 utt, 1.1 h | 760 utt, 0.8 h |
| VietMed | medical talk and phone calls, **8 kHz** (stored at 16 kHz) | 2,773 utt, 4.8 h | 2,912 utt, 5.0 h | 1,000 of 3,436 utt |

- VIVOS has no official dev split: dev is 2 male + 2 female speakers held out of train.
- VietMed test is a fixed subset of 1,000 utterances, stratified by `rec_condition`. Only
  these are in the test plan.
- Utterances shorter than 0.5 s or longer than 30 s were dropped.

### Noise

| Pool | Source | train | test |
|---|---|---|---|
| Traffic | FSD50K road-traffic clips, no speech or music, ≥ 3 s, ≤ 10% digital silence | 1,115 clips, 4.3 h | 368 clips, 1.5 h |
| Voices | MUSAN `speech/librivox`, one voice per file, balanced by gender | 120 voices, 14.0 h | 52 voices, 6.3 h |

Babble is not stored as files: it is the sum of 3–5 voices, made when a sample is mixed.
Train noise and test noise come from different files: test noise is never heard in
training or dev.

### Conditions

| Condition | Meaning |
|---|---|
| `clean` | The clean utterance |
| `traffic_20dB` … `traffic_0dB` | Traffic noise at SNR 20, 10, 5, 0 dB |
| `babble_20dB` … `babble_0dB` | Babble at SNR 20, 10, 5, 0 dB |

Lower SNR means noisier; at 0 dB the noise is as loud as the speech. For VietMed, the noise
is limited to the same 4 kHz band as the speech.

## 3. Manifests

| File | Rows | One row per | Used for |
|---|---|---|---|
| `clean.csv` | 25,740 | utterance (all splits) | references, clean training audio |
| `test_plan.csv` | 23,490 | test utterance × 9 conditions | final test |
| `dev_plan.csv` | 4,305 | dev utterance, one condition each (25% clean) | checkpoint selection |
| `train_schedule.csv` | 163,890 | train utterance × epoch (10 epochs) | `ft-noisy` training |
| `noise_traffic.csv` | 1,483 | traffic clip | `ft-noisy` training |
| `noise_voices.csv` | 172 | voice | `ft-noisy` training |

Row counts are those of v1; `DATASET.json` holds the counts of the attached version.

**`clean.csv`**

| Column | Meaning |
|---|---|
| `utt_id` | Unique ID, prefixed by dataset: `fleurs_…`, `vivos_…`, `vietmed_…` |
| `dataset`, `split` | `fleurs` / `vivos` / `vietmed`; `train` / `dev` / `test` |
| `path` | `clean/<dataset>/<split>/<utt_id>.wav` |
| `text` | **Normalized transcript: the reference and the training target** |
| `text_raw` | Original transcript, for reading only |
| `duration_s`, `peak`, `has_clipping` | Measured on the 16 kHz audio |
| `orig_sr` | Original sample rate (8000 for VietMed) |
| `speaker`, `gender` | Empty speaker for FLEURS |
| `has_digit` | The original transcript had digits (see `evaluation.md` §7) |
| `sentence_id` | FLEURS only |
| `rec_condition` | VietMed only: Tel, Consultation, News… |

**`test_plan.csv`, `dev_plan.csv`**

| Column | Meaning |
|---|---|
| `utt_id`, `dataset`, `split` | As in `clean.csv` |
| `condition` | `clean`, `traffic_5dB`… |
| `noise_type`, `snr_db` | `clean` / `traffic` / `babble`; target SNR (empty for clean) |
| `path` | **The audio to transcribe**: the noisy file, or the clean file for `clean` |
| `clean_path` | The clean source |
| `noise_id`, `offset_s`, `voices`, `band_sr` | How the noise was made (traffic clip and offset, babble voices) |
| `snr_measured`, `peak`, `has_clipping` | Quality check of the written file |

**`train_schedule.csv`**: `epoch`, `utt_id`, `noise_type`, `snr_db` (uniform in 0–20 dB),
`noise_id`, `offset_s`, `voices`, `band_sr`. One row per train utterance and epoch; 25% of
the rows are clean.

**`noise_*.csv`**: `noise_id`, `source`, `split`, `path`, `duration_s`, `gender`, `labels`.

Plans and schedule are fixed tables: every model is tested on the same noisy files and
trained on the same noisy samples.

## 4. Loading data

Everything below returns model-independent data: 16 kHz `float32` arrays and normalized
text. Feature extraction, tokenization and the model itself stay in your notebook.

### Names

All systems use the same names, so their results can be put in one table:

| `MODEL` | `VARIANT` |
|---|---|
| `qwen3-asr-0.6b`, `phowhisper-small` | `zero-shot`, `ft-clean`, `ft-noisy` |

Predictions go to `predictions_<MODEL>_<VARIANT>.csv` in `/kaggle/working`.

### Test (zero-shot or fine-tuned)

```python
from dataprep import iter_eval_batches, load_eval_set
from evaluation import PredictionWriter

eval_set = load_eval_set(MANIFEST_DIR / "test_plan.csv", MANIFEST_DIR / "clean.csv")
# eval_set = load_eval_set(..., limit=20)                    # dry run: 20 utterances per dataset
# eval_set = load_eval_set(..., datasets=["vivos"], conditions=["clean", "babble_0dB"])

with PredictionWriter(f"predictions_{MODEL}_{VARIANT}.csv", MODEL, VARIANT) as writer:
    for rows, audios in iter_eval_batches(writer.pending(eval_set), DATA_ROOT, batch_size=16):
        writer.add(rows, transcribe(audios))     # transcribe: list of arrays -> list of str
```

- `eval_set` has one row per file to transcribe: `utt_id`, `condition`, `dataset`,
  `noise_type`, `snr_db`, `path`, `text`, `duration_s`.
- `iter_eval_batches` yields `(rows, audios)`, sorted by duration to reduce padding.
- `hyp` is written **as the model outputs it**. Do not normalize it; scoring does.
- If the session stops, run the cell again: `pending` skips what is already written.

### Training

```python
import torch
from torch.utils.data import DataLoader
from dataprep import collate_raw, load_train_set

train_set = load_train_set(
    MANIFEST_DIR / "clean.csv", DATA_ROOT, VARIANT,          # "ft-clean" or "ft-noisy"
    schedule=MANIFEST_DIR / "train_schedule.csv",
    noise_pools=[MANIFEST_DIR / "noise_traffic.csv", MANIFEST_DIR / "noise_voices.csv"],
)
# datasets=["vivos"] restricts training to some datasets

def collate(items):                              # model-specific
    batch = collate_raw(items)                   # dict of lists: utt_id, audio, text, noise_type, snr_db
    features = processor(batch["audio"], sampling_rate=16000, return_tensors="pt")
    labels = tokenizer(batch["text"], padding=True, return_tensors="pt").input_ids
    return features, labels

for epoch in range(1, EPOCHS + 1):               # EPOCHS <= 10, the schedule length
    train_set.set_epoch(epoch)
    loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate,
                        num_workers=2, generator=torch.Generator().manual_seed(SEED + epoch))
    for features, labels in loader:
        ...                                      # training step
```

- `train_set[i]` is `{"utt_id", "audio", "text", "noise_type", "snr_db"}`.
- `ft-clean` gives clean audio. `ft-noisy` mixes noise **on the fly** from the schedule row
  of the current epoch, so each epoch gives new noise but every model gets the same samples.
- Call `set_epoch` before each epoch, and keep `persistent_workers=False` (the default) so
  the workers see the new epoch.
- Train only on `split == "train"`, which `load_train_set` does. Never train on dev or test.

### Dev evaluation (checkpoint selection)

```python
import pandas as pd
from dataprep import iter_eval_batches, load_eval_set
from evaluation import score, summarize

dev_plan = pd.read_csv(MANIFEST_DIR / "dev_plan.csv")
clean = pd.read_csv(MANIFEST_DIR / "clean.csv")
dev_set = load_eval_set(dev_plan, clean)                     # 4,305 rows; limit=200 for a quick check

def dev_wer():
    rows, hyps = [], []
    for batch_rows, audios in iter_eval_batches(dev_set, DATA_ROOT, batch_size=16):
        rows.append(batch_rows)
        hyps += transcribe(audios)
    predictions = pd.concat(rows).assign(model=MODEL, variant=VARIANT, hyp=hyps)
    scored = score(predictions[["utt_id", "condition", "model", "variant", "hyp"]], dev_plan, clean)
    return summarize(scored, by=["model", "variant"])["wer"].iloc[0]
```

Call `dev_wer()` after each epoch and keep the checkpoint with the lowest WER. Use the same
rule for every model and variant. The test set is used only once, at the end.

### Inference on any audio

For a demo or a file outside the dataset:

```python
from core import load_audio, mix
from dataprep import take_segment

audio = load_audio("my_recording.wav")          # any format soundfile reads -> 16 kHz mono float32
print(transcribe([audio])[0])

traffic = pd.read_csv(MANIFEST_DIR / "noise_traffic.csv")
noise = load_audio(DATA_ROOT / traffic.query("split == 'test'")["path"].iloc[0])
noisy = mix(audio, take_segment(noise, 0.0, len(audio)), snr_db=5)   # loops a short clip
print(transcribe([noisy])[0])
```

To listen to a test sample in the notebook:

```python
from IPython.display import Audio
row = pd.read_csv(MANIFEST_DIR / "test_plan.csv").query("condition == 'babble_5dB'").iloc[0]
Audio(str(DATA_ROOT / row["path"]))
```

### Scoring

After the test runs, scoring is done in one place for all systems:

```python
from pathlib import Path
from evaluation import load_predictions, missing_predictions, score, wer_table

plan = pd.read_csv(MANIFEST_DIR / "test_plan.csv")
clean = pd.read_csv(MANIFEST_DIR / "clean.csv")
predictions = load_predictions(Path("predictions").glob("predictions_*.csv"))
assert missing_predictions(predictions, plan).empty
table = wer_table(score(predictions, plan, clean))   # dataset × noise × SNR, one column per system
```

WER and CER are in percent, corpus-level, after the same normalization on both sides.
Bootstrap intervals and comparisons are in `evaluation.md`.

## 5. Rules

- Read the data only through `find_data_root()`; never hard-code `/kaggle/input/...`.
- Do not write into the dataset folder (it is read-only on Kaggle). Outputs go to
  `/kaggle/working`.
- Use `text` from `clean.csv` as the training target; do not normalize it again.
- Record `describe(info)` with every result, so results can be traced to a dataset version.
- If a dataset version changes, everyone moves to the new version and reruns.

# Data preparation (`dataprep`)

`dataprep` turns the raw datasets into everything the experiments need: clean speech with
fixed splits, noise pools, noisy test and dev sets, and the noise schedule used to fine-tune
the models. It builds on the audio operations of `core` (see `contract.md`) and adds what
`core` deliberately leaves out: downloads, randomness, manifests and splits.

All randomness takes a `seed`, and every random choice is written to a table. Any noisy file,
and any noisy training sample, can therefore be rebuilt exactly from its table row.

## 1. Where it fits

```
            dataprep                                  notebooks (GPU)          evaluation
 ┌──────────────────────────────────────────────┐   ┌───────────────────┐   ┌──────────────┐
 │ 1 speech  → clean WAV + manifests            │   │ 4 zero-shot runs  │   │              │
 │ 2 noise   → traffic / voice pools            │──▶│ 5 fine-tuning     │──▶│ 6 WER / CER  │
 │ 3 plans   → noisy test + dev, train schedule │   │   (render online) │   │   bootstrap  │
 └──────────────────────────────────────────────┘   └───────────────────┘   └──────────────┘
```

Steps 1–3 run once, on CPU, in notebook `01_prepare_dataset`. Their output is saved as one
Kaggle Dataset, `asr-noisy-vi`, shared with the team (section 5). Every model and every
variant uses it, so all systems are trained and tested on exactly the same data.

| Module | Step | Purpose |
|---|---|---|
| `text.py` | 1, 6 | Transcript normalization, shared with `evaluation` |
| `speech.py` | 1 | Download and standardize FLEURS, VIVOS, VietMed; manifests; splits |
| `noise.py` | 2 | Download FSD50K traffic and MUSAN voices, pools split train/test, copy into the dataset; noise helpers |
| `plan.py` | 3 | Decide every mixing choice: test grid, dev set, training schedule |
| `render.py` | 3, 5 | Turn a plan row into noisy audio; write noisy sets with QC |
| `pipeline.py` | 4, 5 | The shared dataset: find it, check it is complete, `DATASET.json`, `LICENSES.md` |
| `progress.py` | all | One progress-line format for every long step |
| `loaders.py` | 4, 5 | Model-independent data for the notebooks: evaluation batches, training dataset |

### Progress lines

Every long step prints one line about once a minute, in the same format, plus a closing line:

```
FLEURS train: 1250/2994 utterances (41.8%), 21.3 utterances/s, ~2 min left, kept 1247/1250
VIVOS extract: 0.90/1.47 GB (61.2%), 12.4 MB/s, ~1 min left
FSD50K: 640/1818 clips (35.2%), 12.0 clips/s, ~2 min left
noisy test vietmed: 3100/8000 files (38.8%), 35.6 files/s, ~3 min left, max SNR error 0.21 dB, clipped 0
MUSAN: 2.31/11.09 GB (20.8%), 1.9 MB/s, at most 77 min left, reading music/jamendo, voices 0/172
predictions qwen3-asr-0.6b zero-shot: 4800/22374 rows (21.5%), 6.2 rows/s, ~47 min left, 0 already written before
```

| Step | Line |
|---|---|
| `prepare_fleurs`, `prepare_vietmed` | Per split: utterances processed, kept so far |
| `prepare_vivos` | Archive extraction (GB), then utterances |
| `download_fsd50k_traffic` | Clips downloaded (Hugging Face's per-file bars are turned off) |
| `download_musan_speech` | GB streamed, archive folder, voices found |
| `materialize_pool` | Noise files copied |
| `write_noisy_set` | Files written, largest SNR error and clipped files so far |
| `evaluation.PredictionWriter` | Rows written against the rows pending |

`set_progress_interval(seconds)` sets the interval for the whole session (default 60 s,
`None` turns lines off); the notebook sets it from `PROGRESS_INTERVAL_S`. `Progress` and
`ProgressFile` are what the steps use, for any new long loop or download.

## 2. Step 1: clean speech (`speech.py`, `text.py`)

| Dataset | Source | Audio | Splits used |
|---|---|---|---|
| FLEURS `vi_vn` | `google/fleurs`, parquet branch | 16 kHz read speech | train / dev (= validation) / test |
| VIVOS | `AILAB-VNUHCM/vivos`, `vivos.tar.gz` | 16 kHz read speech, quiet room | train / test; **dev = 2 male + 2 female train speakers held out** |
| VietMed | `leduckhai/VietMed`, parquet | **8 kHz** OGG, medical talk and phone calls | train / dev / test |

The files are downloaded directly with `huggingface_hub`, not with `datasets`: recent
`datasets` versions no longer run the VIVOS loading script and need `torchcodec` to decode
audio.

`prepare_fleurs`, `prepare_vivos` and `prepare_vietmed` convert each utterance to 16 kHz
mono with `core.convert` and pass it to `process_utterance`, which:

1. normalizes the transcript (`normalize_text`);
2. drops the utterance if the audio is empty, the text is empty, or the duration is outside
   0.5–30 s, recording the reason;
3. writes kept audio to `clean/<dataset>/<split>/<utt_id>.wav` (PCM_16) and returns a row.

`save_manifests` writes `manifests/clean_<dataset>.csv` (kept) and `dropped_<dataset>.csv`.

### Clean manifest columns

| Column | Meaning |
|---|---|
| `utt_id` | Unique ID, prefixed by dataset (`fleurs_…`, `vivos_…`, `vietmed_…`) |
| `dataset`, `split` | `fleurs` / `vivos` / `vietmed`; `train` / `dev` / `test` |
| `path` | WAV path relative to `DATA_ROOT` |
| `duration_s`, `peak`, `has_clipping` | Measured on the 16 kHz audio |
| `orig_sr` | Original sample rate (8000 for VietMed). Used later to band-limit noise |
| `speaker`, `gender` | Empty speaker for FLEURS |
| `text_raw`, `text` | Original and normalized transcript |
| `has_digit` | The **raw** transcript contains digits (numbers were written as digits) |
| `sentence_id` | FLEURS only: the read sentence |
| `rec_condition` | VietMed only: Tel, Consultation, News… |

### Text normalization

`normalize_text` is the single normalization used for training targets **and** for scoring.
It runs, in order:

1. Unicode NFC, lowercase;
2. numbers written as words (`verbalize_numbers`, below);
3. punctuation replaced by spaces, whitespace collapsed;
4. tone placement unified in word-final `oa`, `oe`, `uy`: `hoà` → `hòa`, `khoẻ` → `khỏe`,
   `thuỷ` → `thủy` (`quý` is left alone). Both spellings are correct Vietnamese, so a model
   must not lose points for using the other one;
5. regional variants of number words unified (below).

#### Numbers

The datasets do not write numbers the same way. Share of transcripts containing digits:

| Dataset | train | dev | test | How numbers are written |
|---|---|---|---|---|
| FLEURS | 23.3% | 12.2% | 20.0% | Digits, Vietnamese separators (`2015`, `9.000`, `19.500 km²`, `21 - 20`) |
| VIVOS | 1 utterance | – | 0% | Words |
| VietMed | 0.8% | 1.5% | 0.9% | Mostly words; a few digits (`3 ngày`, `35 tuổi`) and medical codes (`covid19`, `h1n1`) |

Models may also output digits for speech whose reference uses words. Without a common form,
"2011" against "hai nghìn không trăm mười một" counts as 1 substitution and 5 deletions
although the model heard correctly. All numbers are therefore written as words, the way
they are spoken, on both the reference and the hypothesis side. Training targets use the
same form, so a fine-tuned model learns one convention.

`verbalize_number` follows Vietnamese reading rules:

| Digits | Words | Rule |
|---|---|---|
| `21`, `25`, `15` | hai mươi mốt, hai mươi lăm, mười lăm | `mốt` after `mươi`; `lăm` after `mươi` / `mười` |
| `24` | hai mươi bốn | `bốn`, not `tư` |
| `105` | một trăm linh năm | Zero tens read `linh` |
| `2011` | hai nghìn không trăm mười một | Non-leading groups say their hundreds |
| `1000001` | một triệu không trăm linh một | Empty groups are skipped |
| `0912` | không chín một hai | Leading zero: digit by digit |

`verbalize_numbers` finds the numbers in a text:

- Vietnamese separators: `.` groups thousands (`9.000` → chín nghìn), `,` marks decimals
  (`3,5` → ba phẩy năm);
- a unit right after a number is read too: `%` → phần trăm, `km²` → ki lô mét vuông, `km`,
  `m`, `cm`, `mm`, `kg`;
- digits glued to letters are **left as they are**: medical and technical codes (`covid19`,
  `h1n1`, `hba1c`, `sarscov2`) have no single spoken form.

Number-word variants are mapped to one form, only where they are unambiguous:

| Variant | Canonical |
|---|---|
| `ngàn` | `nghìn` |
| `trăm lẻ`, `nghìn lẻ` | `trăm linh`, `nghìn linh` (`số lẻ` is untouched) |
| `nhăm` | `lăm` |
| `mươi tư` | `mươi bốn` (`thứ tư` is untouched) |
| `mươi một` | `mươi mốt` |

On all 25,754 transcripts of the three datasets, 120 still contain digits after
normalization, all in codes (`covid19`, `f0`, `hba1c`, `h5n1`, `5k`…).

`has_digit` is computed on the **raw** transcript, since the normalized one has no digits
left. It marks the utterances whose score depends on number reading.

Not handled, and to be kept in mind when reading results:

- **Other readings of numbers**: years read digit by digit (`hai không một một`), times
  (`8:30` → tám rưỡi), dates, scores. Report FLEURS results with and without `has_digit`
  utterances as a check.
- **`i` / `y` variants** (`kĩ` / `kỹ`, `lí` / `lý`) are not unified.

### Utilities

| Function | Use |
|---|---|
| `summarize_manifest` | Utterances, hours, speakers, sample rates, % with digits, clipping per dataset × split |
| `speaker_overlap` | Leakage check: values of `key` shared between splits (use `key="sentence_id"` for FLEURS) |
| `subsample` | Reproducible stratified subset, e.g. 1000 VietMed test utterances by `rec_condition` |

## 3. Step 2: noise pools (`noise.py`)

A pool is a table of noise files that may be used, each assigned to the **train** or the
**test** split. Training and dev noise only come from train files and test noise only from
test files, so test noise is never heard before the final evaluation.

| Pool | Function | Content | Split |
|---|---|---|---|
| Traffic | `select_fsd50k` | FSD50K clips with a road-traffic label (`FSD50K_TRAFFIC_LABELS`, including `Motorcycle`) and no speech, crowd or music label (`FSD50K_EXCLUDE_LABELS`) | FSD50K `dev` → train, `eval` → test |
| Voices (for babble) | `select_musan_speech` | MUSAN `speech/librivox` files | By file, 30% test, balanced by gender |

Files shorter than 3 s are left out. The broad label `Vehicle` is not used because it also
covers aircraft, trains and boats. `Chatter`, `Crowd` and `Cheering` are excluded besides the
speech labels: FSD50K files them under "Human group actions", not under `Human_voice`, but
they are voices a model could transcribe. This leaves 1,367 train and 451 test clips before
the 3 s filter.

Clips with more than 10% digital silence (`MAX_SILENT_FRACTION`) are left out too.
`silent_fraction` counts the 50 ms frames whose RMS is below 1e-4 (about −80 dBFS), which is
padding, not quiet street ambience. Some clips are mostly padding (up to 88% of a 15 s clip):
a segment cut from that part is silent, so `mix` cannot scale it to an SNR, and a segment
that is partly silent puts all its noise into the rest of the utterance, which is then
noisier than the planned SNR. About half of the traffic clips have no silent frame at all.

MUSAN has no speaker IDs, so each file is treated as one voice. Most voices are English,
which should be noted in the report.

### Downloads

Neither source has to be attached by hand:

| Function | Source | What is downloaded |
|---|---|---|
| `download_fsd50k_traffic(raw_dir)` | Hugging Face mirror `Fhrozen/FSD50k` (one file per clip) | The two ground-truth CSVs and only the selected clips (~1,800), not the ~30 GB release. Returns the pool |
| `download_musan_speech(raw_dir)` | OpenSLR `musan.tar.gz` (11 GB) | The archive is streamed and only `speech/librivox` files are written; streaming stops once all of them (listed in MUSAN's `ANNOTATIONS`) are found. Its progress line shows the GB read, speed, an upper bound of the time left, the archive folder being read and the voices found. Returns the MUSAN root |

The voices come after the music in the archive, so most of it must be read. OpenSLR serves
it at about 2 MB/s on one connection (measured locally), i.e. more than an hour. A copy of
MUSAN attached as a Kaggle input (`MUSAN_ROOT`) avoids the wait.

Both skip files already on disk, so a stopped run resumes. A copy attached as a Kaggle input
can be used instead (`select_fsd50k(root)` for the Zenodo layout, `select_musan_speech(root)`).

### Copy into the dataset

`materialize_pool(pool, data_root, subdir)` copies the pool's files to
`noise/<subdir>/<noise_id>.wav` as 16 kHz PCM_16 and returns the pool with **relative**
paths. The shared dataset is then self-contained: whoever attaches it needs neither FSD50K
nor MUSAN, and paths stay valid wherever Kaggle mounts it. Traffic clips (44.1 kHz) are
resampled once here instead of at every training step. `NoiseBank(..., data_root=...)`
resolves the relative paths.

### Pool columns

`noise_id`, `source`, `split`, `path` (relative to the dataset root), `duration_s`, `gender`
(voices), `labels` (FSD50K labels or MUSAN language).

### Helpers

| Function | Purpose |
|---|---|
| `take_segment(noise, offset_s, n)` | `n` samples from `offset_s`, looping the noise if it is shorter than the speech |
| `read_segment(path, offset_s, n)` | Same result as `take_segment(load_audio(path), offset_s, n)`, but reads only that part of the file from disk |
| `random_offset_s(rng, noise_s, speech_s)` | A start offset where the speech fits without looping, when possible |
| `band_limit(audio, band_sr)` | Remove content above `band_sr / 2`, by resampling down and back up |

**Why `band_limit`.** VietMed speech was recorded at 8 kHz, so it has nothing above 4 kHz,
while FSD50K and MUSAN noise do. Mixed as is, the noise above 4 kHz would sit where there is
no speech: the model could ignore it easily, and the SNR would not mean the same as for the
other datasets. For utterances whose `orig_sr` is below 16 kHz, the noise is band-limited to
the same band, as if both had gone through the same phone line.

## 4. Step 3: plans (`plan.py`) and rendering (`render.py`)

### The plan row

Every mixing decision is a row (a *spec*); `render(speech, spec, bank)` rebuilds the audio
from it.

| Column | Meaning |
|---|---|
| `noise_type` | `clean`, `traffic` or `babble` |
| `snr_db` | Target SNR; empty for clean |
| `noise_id`, `offset_s` | Traffic clip and start offset |
| `voices` | Babble: `noise_id@offset_s;noise_id@offset_s;…`, 3–5 distinct voices |
| `band_sr` | Set when the speech is narrow-band (VietMed: 8000) |

Test and dev plans also have `utt_id`, `dataset`, `split`, `condition` (`clean`,
`traffic_5dB`…), `clean_path` and `path` (where the noisy file is written).

### Three plans

| Plan | Function | Content | Noise split | Audio |
|---|---|---|---|---|
| Test | `build_test_plan` | Full grid: each utterance × (clean + 2 noise types × SNR 20/10/5/0) = 9 conditions | test | Written once (`write_noisy_set`) |
| Dev | `build_dev_plan` | One condition per utterance (25% clean), SNR from the test levels | train | Written once |
| Train schedule | `build_train_schedule` | One condition per utterance **per epoch** (25% clean), SNR uniform in [0, 20] dB | train | Not written: rendered online |

Design choices:

- **Same noise segment at every SNR.** In the test plan, an utterance gets one traffic
  segment and one babble mix, reused at all four SNR levels. Differences between SNR levels
  then come from the SNR alone.
- **Continuous training SNR.** Training never uses exactly the test levels, so a model
  cannot fit to them.
- **A schedule, not pre-mixed files.** Pre-mixing every training utterance at every
  condition would multiply storage and epoch time by 9, and the model would see the same 8
  noisy copies at every epoch. The schedule gives a new noise draw at every epoch and costs
  no disk. Because it is a fixed table, **every model is trained on exactly the same noisy
  samples**.
- **Clean share.** 25% of training and dev samples stay clean, so fine-tuning does not hurt
  accuracy on clean speech.

### Rendering

`NoiseBank(*pools)` gives access to noise files by `noise_id`. `render(speech, spec, bank)`:

1. clean spec → returns the speech unchanged;
2. traffic → `take_segment` of the clip; babble → `read_segment` of each voice, summed with
   `core.make_babble`;
3. narrow-band speech → `band_limit` of the noise;
4. `core.mix(speech, noise, snr_db)`.

The same speech and spec always give the same audio.

**Memory.** The two noise types are read differently because their files differ in length:

| Noise | File length | Access | Memory |
|---|---|---|---|
| Traffic (FSD50K) | ≤ 30 s | `bank.load`: whole file, kept in an LRU cache (256 files) | ≤ ~0.5 GB |
| Babble voices (MUSAN) | minutes (~7 min on average) | `bank.segment`: only the needed seconds, read from disk, not cached | One utterance long |

Loading MUSAN voices whole would keep the full pool in RAM (~175 files × ~27 MB ≈ 4.7 GB),
once per DataLoader worker, which exhausts Colab's RAM. Reading a 6 s segment takes ~0.2 ms
(measured locally), far below a training step, and the operating system's page cache keeps
frequently read files in memory, shared by all workers. Rendering is therefore safe with
`num_workers > 0`.

`write_noisy_set(plan, bank, data_root)` renders each noisy row of a test or dev plan, writes
it as PCM_16, and re-reads the written file to add `snr_measured`, `peak` and `has_clipping`.
QC: `|snr_measured − snr_db| < 0.5` dB and no clipping. Existing files are not rewritten,
so an interrupted run can resume.

### Sizes

| Dataset | Test utterances | Noisy test files (× 8) |
|---|---|---|
| FLEURS | ~857 | ~6.9k |
| VIVOS | ~760 | ~6.1k |
| VietMed | ~3437, or ~1000 after `subsample` | ~27.5k, or ~8k |

## 5. The shared dataset (`pipeline.py`, notebook `01_prepare_dataset`)

The notebook runs steps 1–3 and writes everything under `/kaggle/working/asr-data`.
Downloads go to `/tmp`, so they are not saved with the output.

**Code on Kaggle.** Code changes go through PRs, so Kaggle does not clone the repo. The code
is uploaded as a second private Kaggle Dataset, `asr-noisy-vi-code`:

1. locally, `python scripts/package_code.py` writes `dist/asr-noisy-vi-code.zip`: the three
   packages, `pyproject.toml`, `README.md`, `LICENSE` and `CODE_VERSION.json` (commit, branch,
   uncommitted changes). Commit first: the script warns when the bundle differs from the
   commit;
2. upload the zip as a new version of `asr-noisy-vi-code`;
3. the notebook finds it under `/kaggle/input`, copies it to `/tmp` (inputs are read-only)
   and installs it with `pip install -e ".[notebooks]"`.

`code_version()` reads `CODE_VERSION.json` (or git, when run from a checkout) and
`write_dataset_info` stores it in `DATASET.json`, so every dataset version names the code that
built it. Model notebooks attach both datasets and install the code the same way.

```
asr-data/
├── DATASET.json        version, code commit, parameters, manifest row counts
├── LICENSES.md         sources and licenses
├── manifests/          clean.csv, noise_traffic.csv, noise_voices.csv,
│                       test_plan.csv, dev_plan.csv, train_schedule.csv
├── clean/<dataset>/<split>/          every split, clean
├── noisy/<dataset>/<split>/<condition>/   test and dev only, mixed in advance
└── noise/{traffic,voices}/            noise used by the plans and the training schedule
```

- `clean/` keeps **all** splits clean: train for fine-tuning, dev and test for the clean
  condition.
- `noisy/` exists only for dev and test. Training audio is never written: `TrainSet` mixes
  `clean/` with `noise/` online, following `train_schedule.csv`.
- `noise/` holds both pool splits: training draws from the train files, `noisy/test` was
  made from the test files.

All paths in the manifests are relative to `asr-data/`.

**Saving and sharing.** After the run: notebook version → *Output* → *New Dataset*, named
`asr-noisy-vi`, **private** (VIVOS and some FSD50K clips are non-commercial, see
`LICENSES.md`), shared with the team. To change a parameter, rerun the notebook and save a
new version of the same dataset; everyone must use the same version.

**Using it.** A model notebook attaches the dataset and never builds data:

```python
from dataprep import describe, find_data_root, require_data

DATA_ROOT = find_data_root()     # $ASR_DATA_ROOT, any /kaggle/input/**/DATASET.json, local folders
info = require_data(DATA_ROOT, steps=("speech", "plans"))         # evaluation
# info = require_data(DATA_ROOT)                                   # ft-noisy also needs "noise"
print(describe(info))            # log the dataset version with the results
MANIFEST_DIR = DATA_ROOT / "manifests"
```

| Function | Purpose |
|---|---|
| `find_data_root(search=None)` | First folder with a `DATASET.json`. If none: an error saying to attach `asr-noisy-vi` or run `01_prepare_dataset` |
| `require_data(root, steps, check_audio=False)` | Checks the manifests the steps need exist and have the row counts of `DATASET.json` (detects an incomplete copy); `check_audio=True` also checks every audio file |
| `write_dataset_info(root, config, version)` | Writes `DATASET.json` (including `code_version()`). Written last, so its presence marks a complete dataset |
| `code_version(code_root=None)` | Commit, branch and uncommitted changes of the running code, from `CODE_VERSION.json` or git |
| `write_licenses(root)` | Writes `LICENSES.md` |
| `missing_audio(root)` | Audio files referenced by the manifests but absent |
| `describe(info)` | One line naming the version, for logs |

## 6. Data for the model notebooks (`loaders.py`)

Steps 4 and 5 (zero-shot runs, fine-tuning, testing) get their data from `loaders.py`. Its
output is the same for every model: 16 kHz float32 audio arrays and normalized transcripts.
A model notebook only adds what is specific to the model.

| Shared (`dataprep`, `evaluation`) | Model-specific (notebook) |
|---|---|
| Which utterances and conditions, reference text, audio loading, batching by duration | Feature extraction (processor) |
| Clean or noisy training samples, per epoch, from the shared schedule | Tokenization of `text`, padding of labels |
| Writing predictions, resuming an interrupted run | `transcribe(audios) -> list[str]`, training loop |

| Function | Use |
|---|---|
| `load_eval_set(plan, manifest, datasets=None, conditions=None, limit=None)` | Rows to transcribe for `test_plan.csv` or `dev_plan.csv`, with `text` and `duration_s`. `limit=20` keeps 20 utterances per dataset for a dry run |
| `iter_eval_batches(eval_set, data_root, batch_size=16)` | `(rows, audios)` batches, sorted by duration to reduce padding |
| `load_train_set(manifest, data_root, variant, schedule, noise_pools)` | `TrainSet` for `"ft-clean"` (clean audio) or `"ft-noisy"` (rendered from the schedule) |
| `TrainSet` | `len`, `[i]` → `{"utt_id", "audio", "text", "noise_type", "snr_db"}`; `set_epoch(n)` |
| `collate_raw(items)` | Batch of items → dict of lists, input of a model's processor |

`TrainSet` does not import PyTorch but works with `torch.utils.data.DataLoader`, including
with several workers (it is picklable, and babble voices are read without a cache). Call
`set_epoch` before each epoch and keep `persistent_workers=False`, so that workers see the
new epoch.

### Zero-shot run or test of a fine-tuned model

```python
from dataprep import iter_eval_batches, load_eval_set
from evaluation import PredictionWriter

eval_set = load_eval_set(MANIFEST_DIR / "test_plan.csv", MANIFEST_DIR / "clean.csv")
with PredictionWriter(OUT_DIR / f"predictions_{MODEL}_{VARIANT}.csv", MODEL, VARIANT) as writer:
    for rows, audios in iter_eval_batches(writer.pending(eval_set), DATA_ROOT, batch_size=16):
        writer.add(rows, transcribe(audios))        # model-specific
```

Run again after an interruption: `pending` skips what is already written.

### Fine-tuning

```python
from torch.utils.data import DataLoader
from dataprep import collate_raw, load_train_set

train_set = load_train_set(
    MANIFEST_DIR / "clean.csv", DATA_ROOT, VARIANT,          # "ft-clean" or "ft-noisy"
    schedule=MANIFEST_DIR / "train_schedule.csv",
    noise_pools=[MANIFEST_DIR / "noise_traffic.csv", MANIFEST_DIR / "noise_voices.csv"],
)
dev_set = load_eval_set(MANIFEST_DIR / "dev_plan.csv", MANIFEST_DIR / "clean.csv")

def collate(items):                                         # model-specific
    batch = collate_raw(items)
    features = processor(batch["audio"], sampling_rate=16000, return_tensors="pt")
    labels = tokenizer(batch["text"], padding=True, return_tensors="pt").input_ids
    return features, labels

for epoch in range(1, EPOCHS + 1):
    train_set.set_epoch(epoch)
    for features, labels in DataLoader(train_set, batch_size=8, shuffle=True,
                                       collate_fn=collate, num_workers=2):
        ...                                                 # training step
    # choose the checkpoint on dev_set with iter_eval_batches + evaluation.score
```

Both variants use the same code; only `VARIANT` changes. For identical sample order
across models, seed the shuffle (`generator=torch.Generator().manual_seed(SEED + epoch)`).

## 7. For the report

- Noise is simulated by additive mixing; there is no reverberation, channel distortion or
  Lombard effect.
- SNR is computed over the whole utterance, silences included (see `contract.md`).
- Traffic noise: FSD50K road-traffic classes, with no speech or music labels. FSD50K licenses
  differ per clip.
- Babble: 3–5 MUSAN Librivox voices, mostly English, balanced by RMS.
- Train and test noise come from disjoint files.
- VietMed is 8 kHz and conversational: it differs from the other two datasets in channel and
  domain, not only in noise. Its noise is band-limited to 4 kHz.

# CLAUDE.md

## Project

Vietnamese ASR under noise. Measures how much zero-shot ASR models (Qwen3-ASR-0.6B, PhoWhisper-small) degrade when noise (traffic, babble) is added at SNR levels of 20/10/5/0 dB, and whether LoRA fine-tuning with online noise mixing helps. Speech: FLEURS `vi_vn`, VIVOS, VietMed. Noise: FSD50K (traffic), MUSAN (babble).

The repo has three packages; experiment notebooks call into them:

| Package | Role | Doc |
|---|---|---|
| `core/` | Audio operations: convert, mix, measure | `docs/contract.md` |
| `dataprep/` | Datasets, noise pools, mixing plans, noisy sets, training schedule | `docs/dataprep.md` |
| `evaluation/` | WER / CER from model outputs, summary tables, bootstrap | `docs/evaluation.md` |

## Working rules

- **Do nothing that was not asked for.** Do not edit code, create files, commit or push unless the user explicitly asks. While the user is discussing, only answer and propose.
- **Everything in the repo is written in English**: code, comments, docstrings, docs, commit messages. Chat with the user can be in Vietnamese.
  Exception: experiment results and their analysis in `docs/results/` are written in Vietnamese, so the whole team can read them.
- `docs/contract.md` is the source of truth for the core. To change a function's behavior: agree with the user → update the contract → update code and tests.
- `docs/dataprep.md` and `docs/evaluation.md` describe the other two packages; keep them in sync with the code.
- Keep the core small: add a function only if it is an audio operation. Shared experiment logic goes in `dataprep` or `evaluation`; model-specific code stays in notebooks.

## Key concepts

**Audio**: 1-D `np.ndarray`, `float32`, mono, 16 kHz, values in [-1, 1].

**The three core operations** (details in `docs/contract.md`):

| Operation | Functions | Idea |
|---|---|---|
| Convert | `convert`, `load_audio` | Bring any audio to 16 kHz mono float32 |
| Merge | `mix`, `make_babble` | `mix`: add noise to speech at the target SNR; output is as long as the speech; `offset_s` sets where to start reading the noise. `make_babble`: sum several RMS-balanced voices |
| Extract info | `extract_info`, `measure_snr` | Info about one file (duration, peak, empty, clipping, SNR when a clean file is given) |

**SNR** (signal-to-noise ratio): `10 · log10(P(speech) / P(noise))`, with `P = mean(x²)` over the whole utterance. Lower SNR means noisier. `mix` scales only the noise and leaves speech unchanged.

**Anti-clipping**: if mixed audio exceeds 0.99, the whole signal is scaled down; the SNR is unchanged.

**Loudness**: no absolute loudness normalization. Only relative loudness is controlled, through SNR.

**Package boundaries**:

- `core`: one signal or one pair per call, no randomness, no knowledge of manifests or models.
- `dataprep`: downloads, transcript normalization, duration filtering, splits, manifests, noise pools (train/test by file), segment selection and noise looping, seeded random choices written to plan tables, `render()` of a plan row (also used online during training).
- `evaluation`: scoring of saved model outputs only; never runs a model. Uses `dataprep.normalize_text` for both references and hypotheses.
- Notebooks: only what is model-specific (processor, tokenizer, `transcribe`, training loop) and plots. Data comes from `dataprep.loaders`, predictions go through `evaluation.PredictionWriter`.

**Reproducibility**: every random choice takes a `seed` and is saved as a table row (test plan, dev plan, training schedule), from which the audio is rebuilt exactly. All models share the same tables.

**Shared dataset**: notebook `01_prepare_dataset` builds everything once into `asr-data/` (relative paths, `DATASET.json` written last), saved as the private Kaggle Dataset `asr-noisy-vi`. Model notebooks call `find_data_root()` and `require_data()` and never build data; if it is missing they stop with an error. `docs/dataset.md` describes its content and how to load it for training, dev, test and inference; keep it in sync with `dataprep` and `evaluation`.

## Code conventions

- Functions never modify their inputs; they accept an `np.ndarray` or a list of numbers.
- One signal or one pair per call, no batching.
- Time is in seconds (`_s` suffix).
- Constants (`SAMPLE_RATE`, `MAX_PEAK`, `CLIP_THRESHOLD`) live in `core/constants.py`.
- Use only basic NumPy APIs so the core runs on any `numpy>=2.0,<3` (Colab and Kaggle ship their own NumPy; notebooks should not reinstall it).

## Layout

| Module | Purpose | Tests |
|---|---|---|
| `core/convert.py` | Convert | `tests/test_convert.py` |
| `core/merge.py` | Merge | `tests/test_merge.py` |
| `core/info.py` | Extract info | `tests/test_info.py` |
| `dataprep/text.py` | Transcript normalization | `tests/test_dataprep_text.py` |
| `dataprep/speech.py` | FLEURS, VIVOS, VietMed → clean WAV + manifests | `tests/test_dataprep_speech.py` |
| `dataprep/noise.py` | Noise pools, looping, band limiting | `tests/test_dataprep_noise.py` |
| `dataprep/plan.py` | Test plan, dev plan, training schedule | `tests/test_dataprep_plan.py` |
| `dataprep/render.py` | Plan row → noisy audio; write noisy sets with QC | `tests/test_dataprep_render.py` |
| `dataprep/pipeline.py` | Shared dataset: find, check, `DATASET.json`, `LICENSES.md` | `tests/test_dataprep_pipeline.py` |
| `dataprep/loaders.py` | Model-independent eval batches and training dataset | `tests/test_dataprep_loaders.py` |
| `dataprep/progress.py` | One progress-line format for long steps (`set_progress_interval`) | `tests/test_dataprep_progress.py` |
| `evaluation/metrics.py` | Edit counts, error rate | `tests/test_evaluation_metrics.py` |
| `evaluation/predictions.py` | Prediction writer with resume | `tests/test_evaluation_predictions.py` |
| `evaluation/scoring.py`, `evaluation/bootstrap.py` | Scoring, tables, bootstrap | `tests/test_evaluation_scoring.py` |

`scripts/package_code.py` bundles the code for Kaggle (`tests/test_scripts_package_code.py`).

Every public function is exported from its package (`from core import mix`, `from dataprep import render`). Each module has a matching test file.

## Setup and import

The packages are `core/`, `dataprep/` and `evaluation/` at the repo root.

All dependencies are declared in `pyproject.toml`, the single place to add or change a library:

| Group | Contents | Install |
|---|---|---|
| core | `numpy`, `soundfile`, `soxr` | `pip install -e .` |
| `notebooks` | core + `pandas`, `pyarrow`, `huggingface_hub`, `rapidfuzz` (needed by `dataprep`, `evaluation` and notebooks) | `pip install -e ".[notebooks]"` |
| `dev` | `notebooks` + `pytest` | `pip install -e ".[dev]"` |

A library used outside `core` goes in the `notebooks` extra, never in the core dependencies.

**Local:**

```bash
git clone https://github.com/anhvinhphan659/asr-noisy-vi
cd asr-noisy-vi
pip install -e ".[notebooks]"   # editable: code edits apply without reinstalling
```

```python
from core import load_audio, mix, extract_info
```

**Kaggle:** code changes go through PRs, so Kaggle does not clone the repo. The code is uploaded as the private Kaggle Dataset `asr-noisy-vi-code`:

```bash
python scripts/package_code.py   # -> dist/asr-noisy-vi-code.zip, with CODE_VERSION.json
```

Upload the zip as a new version of `asr-noisy-vi-code` and attach it to the notebook. The notebook copies it to `/tmp` (inputs are read-only), runs `pip install -e ".[notebooks]"` there, and `DATASET.json` records `CODE_VERSION.json`. Commit before packaging so the bundle maps to a commit.

Do not reinstall NumPy on Colab or Kaggle; use the preinstalled version (`numpy>=2.0,<3` is supported).

**Notebooks** are run and managed outside the repo. A `notebooks/` folder, if present, is only a snapshot: no package imports from it, and it does not need to run in place.

## Commands

```bash
pip install -e ".[dev]"   # with test dependencies
python3 -m pytest
```

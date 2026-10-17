# CLAUDE.md

## Project

Vietnamese ASR under noise. Measures how much a zero-shot ASR model (Qwen3-ASR-0.6B) degrades when noise (traffic, babble) is added at SNR levels of 20/10/5/0 dB, and whether LoRA fine-tuning with online noise mixing helps. Data: FLEURS `vi_vn` (speech), FSD50K (traffic), MUSAN (babble).

The repo contains a small core library (`core/`) for audio processing; experiment notebooks call into it.

## Working rules

- **Do nothing that was not asked for.** Do not edit code, create files, commit or push unless the user explicitly asks. While the user is discussing, only answer and propose.
- **Everything in the repo is written in English**: code, comments, docstrings, docs, commit messages. Chat with the user can be in Vietnamese.
- `docs/contract.md` is the source of truth for the core. To change a function's behavior: agree with the user → update the contract → update code and tests.
- Keep the core small: add a function only if it is an audio operation; orchestration logic stays in notebooks.

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

**Core / notebook boundary**:

- Core: one signal or one pair per call, no randomness, no knowledge of manifests or models.
- Notebooks: segment selection and noise looping, random choices (noise, SNR, offset, babble voices), duration filtering, manifests, `augment()`, running models, WER/CER.

## Code conventions

- Functions never modify their inputs; they accept an `np.ndarray` or a list of numbers.
- One signal or one pair per call, no batching.
- Time is in seconds (`_s` suffix).
- Constants (`SAMPLE_RATE`, `MAX_PEAK`, `CLIP_THRESHOLD`) live in `core/constants.py`.
- Use only basic NumPy APIs so the core runs on any `numpy>=2.0,<3` (Colab and Kaggle ship their own NumPy; notebooks should not reinstall it).

## Layout

| Operation | Module | Tests |
|---|---|---|
| Convert | `core/convert.py` | `tests/test_convert.py` |
| Merge | `core/merge.py` | `tests/test_merge.py` |
| Extract info | `core/info.py` | `tests/test_info.py` |

Every public function is exported from `core` (`from core import mix`). Each module has a matching test file.

## Commands

```bash
pip install -e ".[dev]"
python3 -m pytest
```

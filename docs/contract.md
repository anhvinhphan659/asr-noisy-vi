# Contract: core audio-processing logic

This document defines the functions in `core/`. It is the source of truth: to change a function's behavior, update this contract first, then the code and tests.

## 1. Scope

The core does exactly **three audio operations**:

| Operation | Purpose | Functions |
|---|---|---|
| Convert | Bring audio to 16 kHz mono float32 | `convert`, `load_audio` |
| Merge | Add noise to speech at a target SNR; sum several voices into babble | `mix`, `make_babble` |
| Extract info | Describe an audio file; measure SNR | `extract_info`, `measure_snr` |

Modules: `core/convert.py`, `core/merge.py`, `core/info.py`. All public names are exported from `core`.

**Out of scope** (handled in notebooks):

- Running models: transcription, fine-tuning.
- Segment selection: which part to cut, looping short noise, choosing an offset.
- Random choices: noise type, noise file, SNR level, voices for babble.
- Threshold filtering (utterances > 30 s, noise < 3 s): compare `duration_s` with the threshold.
- Train/test splits, CSV manifests, `augment()` for training.
- Evaluation: text normalization, WER/CER, bootstrap, plots.

## 2. Conventions

| Convention | Rule |
|---|---|
| Audio representation | 1-D `np.ndarray`, `float32`, mono, 16 kHz, values in [-1, 1] |
| Input | Accepts an `np.ndarray` or a list of numbers |
| Unit of work | Each function handles **one signal or one pair**, no batching. Notebooks loop over utterances |
| Inputs are never modified | Functions always return a new array |
| Loudness | No absolute loudness normalization; original levels are kept |
| Time | Always in seconds (`_s` suffix) for parameters and fields |

### Constants (`core/constants.py`)

| Name | Value | Meaning |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Standard sample rate |
| `MAX_PEAK` | 0.99 | Maximum peak of audio returned by `mix` |
| `CLIP_THRESHOLD` | 0.999 | Peak at or above which audio counts as clipped |

### Definitions

| Term | Definition |
|---|---|
| Power `P(x)` | `mean(x²)`, over the whole signal, silences included |
| RMS | `sqrt(P(x))` |
| SNR (dB) | `10 · log10(P(speech) / P(noise))`. Lower SNR means noisier: 20 dB is nearly clean, 0 dB means speech and noise are equally loud |
| Peak | `max(|x|)` |

## 3. Convert

### `convert(audio, sr_in, sr_out=16000) -> np.ndarray`

Converts an in-memory audio array to the standard representation.

| | |
|---|---|
| Input | `audio`: array of shape `(n,)` or `(n, channels)`, floating point. `sr_in`: original sample rate. `sr_out`: target sample rate |
| Output | 1-D `float32` array at `sr_out` |
| Rules | Multiple channels → averaged. `sr_in ≠ sr_out` → resampled (soxr, high quality). `sr_in = sr_out` → a copy is returned |
| Errors | More than 2 dimensions; integer dtype (avoids wrong value scaling) |
| Use when | Audio does not come from a file: FLEURS via HuggingFace `datasets`, uploaded audio in the demo |

### `load_audio(path, sr=16000) -> np.ndarray`

Reads an audio file, then calls `convert`.

| | |
|---|---|
| Input | `path`: file path (wav, flac, ...). `sr`: target sample rate |
| Output | Same as `convert` |
| Errors | File missing or unreadable |

## 4. Merge

### `mix(speech, noise, snr_db, offset_s=0.0, sr=16000) -> np.ndarray`

Adds `noise` to `speech` so that the speech-to-noise ratio equals `snr_db`.

| | |
|---|---|
| Input | `speech`, `noise`: audio at the same `sr`. `snr_db`: target SNR. `offset_s`: where to start reading `noise` (seconds, ≥ 0) |
| Output | Mixed audio, **exactly as long as `speech`** |

Rules:

1. Take the noise segment starting at `offset_s` with the length of `speech`. Any remaining noise after it is ignored.
2. Only the noise is scaled; speech is unchanged: `g = sqrt(P(speech) / (P(noise_seg) · 10^(snr_db/10)))`.
3. `out = speech + g · noise_seg`.
4. If `peak(out) > MAX_PEAK`, multiply **all** of `out` by `MAX_PEAK / peak(out)`. The SNR is unchanged; only the overall level drops.

```
offset_s = 3
speech:  [==========]              0 → 10 s
noise:   [...|..........|.....]    0 → 15 s
              ↑ uses 3 → 13 s, the rest is ignored
```

Errors:

- The noise from `offset_s` onward is shorter than `speech`. Speech is never cut, because cutting would drop words that are still in the transcript. If the noise is short, the notebook must loop it before calling `mix`.
- `offset_s < 0`.
- `speech` or the noise segment is all zeros (SNR is undefined).

Example: speech RMS = 0.1, noise RMS = 0.05.

| `snr_db` | Noise RMS after scaling | `g` |
|---|---|---|
| 20 | 0.010 | 0.20 |
| 10 | 0.032 | 0.63 |
| 5 | 0.056 | 1.12 |
| 0 | 0.100 | 2.00 |

### `make_babble(voices) -> np.ndarray`

Builds babble (crowd) noise by summing several voices.

| | |
|---|---|
| Input | `voices`: list of speech signals (MUSAN), each from a different speaker. The plan uses 3–5 voices |
| Output | Babble audio, as long as the **shortest voice** |

Rules:

1. `L` = length of the shortest voice; each voice contributes its first `L` samples.
2. Each voice is divided by its RMS (brought to RMS = 1) so no talker dominates.
3. `out = sum of voices / number of voices`.
4. No peak normalization: the absolute level does not matter because `mix` rescales the noise. A notebook that saves babble to a file handles the peak itself.

Errors: `voices` is empty; a voice is all zeros.

Which voices and which segments to use is decided in the notebook (e.g. with `rng.choice`).

## 5. Extract info

### `extract_info(path, clean_path=None) -> AudioInfo`

Returns information about one audio file. Audio fields are computed **after** `load_audio` (16 kHz mono), i.e. on the audio the model actually receives.

| Field | Type | Meaning |
|---|---|---|
| `path` | str | File path |
| `orig_sr` | int | Original sample rate of the file |
| `orig_channels` | int | Original number of channels |
| `duration_s` | float | Duration in seconds |
| `peak` | float | `max(|x|)` |
| `is_empty` | bool | Zero length or all zeros (`peak == 0`) |
| `has_clipping` | bool | `peak ≥ CLIP_THRESHOLD`. Also catches PCM_16 files clipped on save (values read back are ≤ 0.99997) |
| `snr_db` | float \| None | Only when `clean_path` is given: `measure_snr(clean, audio)` |

Errors: unreadable file; `clean_path` given but the two files differ in length.

Results can be turned into a table directly:

```python
df = pd.DataFrame([asdict(extract_info(p)) for p in paths])
```

### `measure_snr(clean, noisy) -> float`

Measures the actual SNR (dB) of `noisy` against its clean reference `clean`. Used for QC: the error versus the target SNR must be < 0.5 dB.

| | |
|---|---|
| Input | `clean`, `noisy`: two audio arrays of the same length |
| Output | SNR in dB |
| Rules | Project `clean` onto `noisy`: `a = <noisy, clean> / <clean, clean>`, `SNR = 10 · log10(P(a·clean) / P(noisy − a·clean))`. The projection makes the result unaffected by the anti-clipping scaling in `mix` and by PCM_16 rounding on save |
| Errors | Length mismatch; `clean` is all zeros |

## 6. Example: composing the functions in a notebook

Building the noisy test set (E1), using the same noise segment at every SNR level:

```python
speech = load_audio(fleurs_path)
noise = load_audio(noise_path)
# notebook: loop noise if shorter than speech, pick a random offset
offset_s = rng.uniform(0, (len(noise) - len(speech)) / SAMPLE_RATE)
for snr in [20, 10, 5, 0]:
    noisy = mix(speech, noise, snr, offset_s=offset_s)
    # notebook: save the file, record noise_id, offset_s, snr_db in the manifest
```

QC:

```python
info = extract_info(noisy_path, clean_path=fleurs_path)
assert abs(info.snr_db - snr) < 0.5
assert not info.has_clipping
```

## 7. For the report

- SNR is computed over the whole utterance, silences included.
- No loudness normalization; FLEURS levels are kept as recorded.
- Babble balances voices by whole-segment RMS, so a voice with more silence can stand out slightly.
- Mixed audio above 0.99 is scaled down as a whole to avoid clipping; the SNR is unchanged.

# asr-noisy-vi

Vietnamese speech recognition (ASR) under noisy conditions.

This repo measures how much a Vietnamese ASR model degrades when background noise is added, and whether LoRA fine-tuning on noisy data helps. It is built as a small audio-processing core library, with notebooks and experiment scripts layered on top.

> **Status:** building the core library first. Notebooks, inference, fine-tuning, evaluation and the demo come later.

## Overview

**Research questions**

1. How much does a zero-shot ASR baseline degrade on noisy speech?
2. How does the degradation vary by noise type and SNR level?
3. Does LoRA fine-tuning with online noise augmentation improve robustness without hurting clean-speech accuracy?

**Models**

- Baseline: Qwen3-ASR-0.6B (zero-shot)
- Fallback for fine-tuning: PhoWhisper-small with LoRA

**Data**

| Data | Source | Use |
|---|---|---|
| Vietnamese speech (`vi_vn`) | [FLEURS](https://huggingface.co/datasets/google/fleurs) | Clean speech, reference transcripts |
| Traffic noise | [FSD50K](https://zenodo.org/records/4060432) | Noise (per-clip licenses vary) |
| Speech for babble noise | [MUSAN](https://www.openslr.org/17/) (`speech/`) | Babble noise (3–5 voices summed) |

**Test set:** 500 FLEURS test utterances × 2 noise types × 4 SNR levels (20/10/5/0 dB) = 4,000 noisy files, plus 500 clean files.

**Experiments**

| ID | Model | Test data |
|---|---|---|
| E0 | Zero-shot baseline | Clean |
| E1 | Zero-shot baseline | Noisy (2 types × 4 SNRs) |
| E2 | LoRA fine-tuned | Clean + noisy |
| E3 (optional) | Fine-tuned without 25% clean samples | Clean + noisy |

## Core library

The first milestone is `asr_noisy`, a small library for audio handling.

| Module | Purpose |
|---|---|
| `io` | Load audio, resample to 16 kHz, convert to mono, save WAV |
| `noise` | Fit noise to a target length, build babble noise |
| `mix` | Mix noise into speech at a target SNR |
| `metrics` | Duration, measured SNR, clipping check |

Conventions:

- Audio is a `float32` mono NumPy array at 16 kHz, with values in [-1, 1].
- Functions never modify their inputs.
- Random behavior takes an explicit `np.random.Generator`; the default seed is 42.

### Example

```python
import numpy as np
from asr_noisy.io import load_audio
from asr_noisy.noise import fit_length
from asr_noisy.mix import mix_at_snr
from asr_noisy.metrics import measure_snr, has_clipping

rng = np.random.default_rng(42)

speech = load_audio("speech.wav")
noise = load_audio("traffic.wav")

noise, offset = fit_length(noise, len(speech), rng)
noisy = mix_at_snr(speech, noise, snr_db=5)

print(measure_snr(speech, noisy))  # ~5.0
print(has_clipping(noisy))         # False
```

## Repo layout

```
asr-noisy-vi/
├── src/asr_noisy/        # core library
│   ├── io.py
│   ├── noise.py
│   ├── mix.py
│   ├── metrics.py
│   └── constants.py
├── tests/                # unit tests
├── notebooks/            # data exploration, listening checks (planned)
├── scripts/              # data prep, inference, fine-tune, evaluate (planned)
├── configs/              # one YAML per experiment (planned)
├── data/manifests/       # CSV manifests only; audio is not in Git
├── results/              # predictions and metrics CSVs (planned)
└── demo/                 # Gradio demo (planned)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+.

## Data rules

- Large audio files stay out of Git. Git holds code, manifests and result CSVs.
- Noise is split **by file**, not by segment: 70% train, 30% test, no file in both. This avoids leakage.
- The test set is used only for final reporting, never for choosing hyperparameters.

## Roadmap

- [ ] Core library (`io`, `noise`, `mix`, `metrics`) with unit tests
- [ ] Notebook: data exploration and listening checks
- [ ] Data prep scripts and noisy test set generation
- [ ] Online noise augmentation (`augment()`) for training
- [ ] Inference and evaluation (WER/CER, bootstrap CIs)
- [ ] LoRA fine-tuning
- [ ] Gradio demo and report

## Limitations

- FLEURS is read speech, not spontaneous speech.
- Noise is simulated by mixing, not recorded in real environments.

## Licenses and attribution

FLEURS, MUSAN and FSD50K each have their own licenses. FSD50K licenses differ per clip and are stored in the noise manifest. Cite the datasets and models used in any report.
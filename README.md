# asr-noisy-vi

Vietnamese speech recognition (ASR) under noisy conditions.

This project measures how much a Vietnamese ASR model degrades when background noise is added, and whether LoRA fine-tuning on noisy data helps.

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

## Data rules

- Large audio files stay out of Git. Git holds code, manifests and result CSVs.
- Noise is split **by file**, not by segment: 70% train, 30% test, no file in both. This avoids leakage.
- The test set is used only for final reporting, never for choosing hyperparameters.

## Limitations

- FLEURS is read speech, not spontaneous speech.
- Noise is simulated by mixing, not recorded in real environments.

## Licenses and attribution

FLEURS, MUSAN and FSD50K each have their own licenses. FSD50K licenses differ per clip and are stored in the noise manifest. Cite the datasets and models used in any report.
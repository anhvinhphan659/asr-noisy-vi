from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ._utils import as_audio, power
from .constants import CLIP_THRESHOLD, SAMPLE_RATE
from .convert import load_audio


@dataclass(frozen=True)
class AudioInfo:
    path: str
    orig_sr: int
    orig_channels: int
    duration_s: float
    peak: float
    is_empty: bool
    has_clipping: bool
    snr_db: float | None = None


def extract_info(path, clean_path=None):
    """Describe an audio file after loading it as 16 kHz mono.

    With ``clean_path``, also report the SNR against that clean reference.
    """
    meta = sf.info(path)
    x = load_audio(path)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    snr_db = None
    if clean_path is not None:
        snr_db = measure_snr(load_audio(clean_path), x)
    return AudioInfo(
        path=str(Path(path)),
        orig_sr=int(meta.samplerate),
        orig_channels=int(meta.channels),
        duration_s=len(x) / SAMPLE_RATE,
        peak=peak,
        is_empty=peak == 0.0,
        has_clipping=peak >= CLIP_THRESHOLD,
        snr_db=snr_db,
    )


def measure_snr(clean, noisy):
    """Actual SNR (dB) of ``noisy`` against its clean reference.

    ``clean`` is projected onto ``noisy`` first, so a global gain applied after
    mixing (anti-clipping, PCM rounding) does not affect the result.
    """
    clean = as_audio(clean, "clean").astype(np.float64)
    noisy = as_audio(noisy, "noisy").astype(np.float64)
    if clean.shape != noisy.shape:
        raise ValueError(f"length mismatch: clean {clean.shape} vs noisy {noisy.shape}")
    if power(clean) == 0:
        raise ValueError("clean is silent; SNR is undefined")
    gain = np.dot(noisy, clean) / np.dot(clean, clean)
    target = gain * clean
    p_residual = power(noisy - target)
    if p_residual == 0:
        return float("inf")
    return float(10 * np.log10(power(target) / p_residual))

import numpy as np
import soundfile as sf
import soxr

from ._utils import as_audio
from .constants import SAMPLE_RATE


def convert(audio, sr_in, sr_out=SAMPLE_RATE):
    """Convert in-memory audio to 1-D float32 mono at ``sr_out``.

    ``audio`` is ``(n,)`` or ``(n, channels)``; channels are averaged.
    """
    x = as_audio(audio, max_ndim=2)
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError(f"sample rates must be positive, got {sr_in} and {sr_out}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr_in != sr_out:
        x = soxr.resample(x, sr_in, sr_out, quality="HQ")
    return np.array(x, dtype=np.float32)


def load_audio(path, sr=SAMPLE_RATE):
    """Read an audio file and convert it to 1-D float32 mono at ``sr``."""
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    return convert(x, file_sr, sr)

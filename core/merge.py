import numpy as np

from ._utils import as_audio, power
from .constants import MAX_PEAK, SAMPLE_RATE


def mix(speech, noise, snr_db, offset_s=0.0, sr=SAMPLE_RATE):
    """Add ``noise`` to ``speech`` at ``snr_db``. Output is as long as ``speech``.

    Noise is read from ``offset_s`` for ``len(speech)`` samples; the rest is
    ignored. Only the noise is scaled. If the mixture peaks above
    ``MAX_PEAK``, the whole mixture is scaled down, keeping the SNR.
    """
    speech = as_audio(speech, "speech").astype(np.float64)
    noise = as_audio(noise, "noise").astype(np.float64)
    if offset_s < 0:
        raise ValueError(f"offset_s must be >= 0, got {offset_s}")
    start = int(round(offset_s * sr))
    segment = noise[start : start + len(speech)]
    if len(segment) < len(speech):
        raise ValueError(
            f"noise from offset_s={offset_s} has {len(segment)} samples, "
            f"speech needs {len(speech)}; loop the noise before calling mix()"
        )
    p_speech = power(speech)
    p_noise = power(segment)
    if p_speech == 0:
        raise ValueError("speech is silent; SNR is undefined")
    if p_noise == 0:
        raise ValueError("noise segment is silent; SNR is undefined")
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    out = speech + gain * segment
    peak = np.max(np.abs(out))
    if peak > MAX_PEAK:
        out *= MAX_PEAK / peak
    return out.astype(np.float32)


def make_babble(voices):
    """Sum several voices into babble noise, as long as the shortest voice.

    Each voice is cut to that length and divided by its RMS so no talker
    dominates. The level is not peak-normalized; ``mix`` rescales noise.
    """
    if len(voices) == 0:
        raise ValueError("voices is empty")
    voices = [as_audio(v, f"voices[{i}]").astype(np.float64) for i, v in enumerate(voices)]
    length = min(len(v) for v in voices)
    out = np.zeros(length, dtype=np.float64)
    for i, voice in enumerate(voices):
        segment = voice[:length]
        p = power(segment)
        if p == 0:
            raise ValueError(f"voices[{i}] is silent")
        out += segment / np.sqrt(p)
    return (out / len(voices)).astype(np.float32)

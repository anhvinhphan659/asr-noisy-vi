"""Core audio processing for Vietnamese ASR under noise. See docs/contract.md."""

from .constants import CLIP_THRESHOLD, MAX_PEAK, SAMPLE_RATE
from .convert import convert, load_audio
from .info import AudioInfo, extract_info, measure_snr
from .merge import make_babble, mix

__all__ = [
    "CLIP_THRESHOLD",
    "MAX_PEAK",
    "SAMPLE_RATE",
    "AudioInfo",
    "convert",
    "extract_info",
    "load_audio",
    "make_babble",
    "measure_snr",
    "mix",
]

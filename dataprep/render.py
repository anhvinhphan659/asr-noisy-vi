"""Turn a plan row into noisy audio, for writing test/dev sets or for online training."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core import CLIP_THRESHOLD, SAMPLE_RATE, load_audio, make_babble, measure_snr, mix

from .noise import band_limit, read_segment, take_segment
from .progress import Progress


def _missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def parse_voices(voices):
    """``"id@1.5;id@0.2"`` -> ``[("id", 1.5), ("id", 0.2)]``."""
    out = []
    for item in voices.split(";"):
        noise_id, offset_s = item.rsplit("@", 1)
        out.append((noise_id, float(offset_s)))
    return out


class NoiseBank:
    """Gives access to noise files by ``noise_id``.

    ``load`` reads a whole file as 16 kHz audio and keeps the most recent ones in memory;
    it is meant for short files (traffic clips, at most 30 s). ``segment`` reads only the
    part that is needed, without caching; it is meant for long files (MUSAN voices).

    Relative pool paths (as in the shared dataset) are resolved against ``data_root``.
    """

    def __init__(self, *pools, data_root=None, cache_size=256):
        self.paths = {}
        for pool in pools:
            for noise_id, path in zip(pool["noise_id"], pool["path"]):
                path = Path(path)
                if data_root is not None and not path.is_absolute():
                    path = Path(data_root) / path
                self.paths[noise_id] = path
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def load(self, noise_id):
        if noise_id in self._cache:
            self._cache.move_to_end(noise_id)
            return self._cache[noise_id]
        audio = load_audio(self.paths[noise_id])
        self._cache[noise_id] = audio
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return audio

    def segment(self, noise_id, offset_s, n_samples):
        return read_segment(self.paths[noise_id], offset_s, n_samples)


def noise_segment(spec, n_samples, bank):
    """The noise of a spec, cut (and looped if needed) to ``n_samples``, before SNR scaling."""
    if spec["noise_type"] == "traffic":
        segment = take_segment(bank.load(spec["noise_id"]), spec["offset_s"], n_samples)
    elif spec["noise_type"] == "babble":
        voices = [bank.segment(noise_id, offset_s, n_samples)
                  for noise_id, offset_s in parse_voices(spec["voices"])]
        segment = make_babble(voices)
    else:
        raise ValueError(f"no noise for noise_type {spec['noise_type']!r}")
    if not _missing(spec.get("band_sr")):
        segment = band_limit(segment, int(spec["band_sr"]))
    return segment


def render(speech, spec, bank):
    """Apply one plan row (test, dev or training schedule) to ``speech``.

    Deterministic: the same speech and spec always give the same audio. For training,
    call it in the dataset's ``__getitem__`` with the schedule row of the current epoch.
    """
    if spec["noise_type"] == "clean":
        return np.array(speech, dtype=np.float32)
    return mix(speech, noise_segment(spec, len(speech), bank), float(spec["snr_db"]))


def write_noisy_set(plan, bank, data_root, label="noisy"):
    """Write every noisy row of a test/dev plan as 16 kHz PCM_16 and run the SNR QC.

    Returns the plan with ``snr_measured``, ``peak`` and ``has_clipping`` added (clean
    rows keep them empty), measured on the written files. Existing files are not
    rewritten, only measured again. Progress lines (named ``label``) show the largest SNR
    error and the clipped files so far.
    """
    data_root = Path(data_root)
    plan = plan.reset_index(drop=True).copy()
    measured = pd.DataFrame(index=plan.index, columns=["snr_measured", "peak", "has_clipping"],
                            dtype=object)
    qc = {"max_error": 0.0, "clipped": 0}

    def status():
        return f"max SNR error {qc['max_error']:.2f} dB, clipped {qc['clipped']}"

    n_noisy = int((plan["noise_type"] != "clean").sum())
    with Progress(label, n_noisy, "files", status=status) as progress:
        for clean_path, rows in plan.groupby("clean_path", sort=False):
            clean = None
            for i, spec in rows.iterrows():
                if spec["noise_type"] == "clean":
                    continue
                if clean is None:
                    clean = load_audio(data_root / clean_path)
                out = data_root / spec["path"]
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, render(clean, spec, bank), SAMPLE_RATE, subtype="PCM_16")
                noisy = load_audio(out)  # QC runs on the file the model will read
                peak = float(np.max(np.abs(noisy)))
                snr = measure_snr(clean, noisy)
                measured.loc[i] = [round(snr, 3), round(peak, 5), peak >= CLIP_THRESHOLD]
                qc["max_error"] = max(qc["max_error"], abs(snr - float(spec["snr_db"])))
                qc["clipped"] += peak >= CLIP_THRESHOLD
                progress.update()
    return pd.concat([plan, measured], axis=1)

"""Noise pools: which noise files may be used, and for which split (train or test)."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core import SAMPLE_RATE, convert, load_audio

from .progress import Progress, ProgressFile

MIN_NOISE_S = 3.0
# Some FSD50K clips are padded with digital silence (up to 88% of a traffic clip). A
# segment cut from such a part is silent and cannot be scaled to an SNR, and a partly
# silent segment puts all its noise into the rest of the utterance. Clips with more than
# this fraction of silent frames are left out of the pool.
MAX_SILENT_FRACTION = 0.1
SILENCE_RMS = 1e-4  # about -80 dBFS: digital silence, not quiet street ambience
SILENCE_FRAME_S = 0.05

# Road traffic classes of the FSD50K vocabulary. "Vehicle" is left out on purpose: it
# also covers aircraft, rail and boats. Motorcycles matter for Vietnamese street noise.
FSD50K_TRAFFIC_LABELS = (
    "Traffic_noise_and_roadway_noise",
    "Motor_vehicle_(road)",
    "Car",
    "Car_passing_by",
    "Motorcycle",
    "Bus",
    "Truck",
    "Vehicle_horn_and_car_horn_and_honking",
)
# Clips with any of these labels are rejected: speech in traffic noise would be
# transcribed by the model, and music is a different noise type. "Chatter", "Crowd" and
# "Cheering" are human voices filed under "Human group actions", not under "Human_voice".
FSD50K_EXCLUDE_LABELS = (
    "Speech",
    "Human_voice",
    "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking",
    "Conversation",
    "Speech_synthesizer",
    "Chatter",
    "Crowd",
    "Cheering",
    "Music",
)

POOL_COLUMNS = ["noise_id", "source", "split", "path", "duration_s", "gender", "labels"]


def _duration_s(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def silent_fraction(path, frame_s=SILENCE_FRAME_S, threshold=SILENCE_RMS):
    """Fraction of ``frame_s`` frames of a file whose RMS is below ``threshold``."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    n = max(1, int(frame_s * sr))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    if len(frames) == 0:
        return 1.0
    return float((np.sqrt((frames ** 2).mean(axis=1)) < threshold).mean())


# Folder names of the original Zenodo release, and of the Hugging Face mirror used by
# download_fsd50k_traffic (one file per clip, so only the selected clips are downloaded).
FSD50K_ZENODO_LAYOUT = {"ground_truth_dir": "FSD50K.ground_truth",
                        "dev_audio_dir": "FSD50K.dev_audio",
                        "eval_audio_dir": "FSD50K.eval_audio"}
FSD50K_HF_REPO = "Fhrozen/FSD50k"
FSD50K_HF_LAYOUT = {"ground_truth_dir": "labels",
                    "dev_audio_dir": "clips/dev",
                    "eval_audio_dir": "clips/eval"}

MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_ANNOTATIONS_REPO = "FluidInference/musan"  # has MUSAN's ANNOTATIONS files, not its audio


def _is_selected(labels, include, exclude):
    label_set = set(labels.split(","))
    return bool(label_set & include) and not label_set & exclude


def select_fsd50k(fsd50k_root, include=FSD50K_TRAFFIC_LABELS, exclude=FSD50K_EXCLUDE_LABELS,
                  min_duration_s=MIN_NOISE_S, max_silent_fraction=MAX_SILENT_FRACTION,
                  ground_truth_dir="FSD50K.ground_truth", dev_audio_dir="FSD50K.dev_audio",
                  eval_audio_dir="FSD50K.eval_audio"):
    """Traffic noise pool from FSD50K.

    The official FSD50K split decides the noise split: clips of ``dev`` (train + val) are
    used for training, clips of ``eval`` for testing, so no clip is in both.
    ``fsd50k_root`` holds the ground-truth CSVs and the two audio folders; their names
    default to the Zenodo release (pass ``**FSD50K_HF_LAYOUT`` for the Hugging Face mirror).
    Clips listed in the ground truth but missing on disk are skipped, and so are clips
    shorter than ``min_duration_s`` or with more than ``max_silent_fraction`` of digital
    silence (``silent_fraction``).
    """
    root = Path(fsd50k_root)
    include, exclude = set(include), set(exclude)
    rows = []
    for gt_name, audio_dir, split in [("dev.csv", dev_audio_dir, "train"),
                                      ("eval.csv", eval_audio_dir, "test")]:
        gt = pd.read_csv(root / ground_truth_dir / gt_name)
        for fname, labels in zip(gt["fname"], gt["labels"]):
            if not _is_selected(labels, include, exclude):
                continue
            path = root / audio_dir / f"{fname}.wav"
            if not path.exists():
                continue
            rows.append({"noise_id": f"fsd50k_{fname}", "source": "fsd50k", "split": split,
                         "path": str(path), "duration_s": round(_duration_s(path), 3),
                         "gender": None, "labels": labels})
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS)
    pool = pool[pool["duration_s"] >= min_duration_s].reset_index(drop=True)
    silent = np.array([silent_fraction(path) for path in pool["path"]])
    keep = silent <= max_silent_fraction
    print(f"FSD50K: {int((~keep).sum())} of {len(pool)} clips dropped, more than "
          f"{max_silent_fraction:.0%} digital silence", flush=True)
    return pool[keep].reset_index(drop=True)


def select_musan_speech(musan_root, subsets=("librivox",), test_fraction=0.3, seed=42,
                        min_duration_s=MIN_NOISE_S):
    """Voice pool for babble from MUSAN ``speech/``.

    MUSAN gives no speaker IDs, so each file is treated as one voice and the split is made
    by file: ``test_fraction`` of the files (balanced by gender) go to test, the rest to
    train. ``ANNOTATIONS`` (file, gender, language) is read when present.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subset in subsets:
        folder = Path(musan_root) / "speech" / subset
        meta = {}
        annotations = folder / "ANNOTATIONS"
        if annotations.exists():
            for line in annotations.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    meta[parts[0]] = (parts[1], " ".join(parts[2:]))
        for path in sorted(folder.glob("*.wav")):
            gender, language = meta.get(path.stem, (None, None))
            rows.append({"noise_id": f"musan_{path.stem}", "source": f"musan_{subset}",
                         "split": None, "path": str(path),
                         "duration_s": round(_duration_s(path), 3),
                         "gender": {"m": "male", "f": "female"}.get(gender, gender),
                         "labels": language})
    pool = pd.DataFrame(rows, columns=POOL_COLUMNS)
    pool = pool[pool["duration_s"] >= min_duration_s].reset_index(drop=True)
    pool["split"] = "train"
    for _, group in pool.groupby(pool["gender"].fillna("unknown")):
        n_test = int(round(test_fraction * len(group)))
        test_idx = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        pool.loc[test_idx, "split"] = "test"
    return pool


# --- Downloads ----------------------------------------------------------------------------

def download_fsd50k_traffic(raw_dir, include=FSD50K_TRAFFIC_LABELS, exclude=FSD50K_EXCLUDE_LABELS,
                            limit=None, workers=8):
    """Download only the FSD50K clips that ``select_fsd50k`` would keep, and return the pool.

    Uses the Hugging Face mirror, which stores one file per clip: about 1,900 clips instead
    of the ~30 GB release. ``limit`` caps the clips per split (dry runs). Clips already on
    disk are not downloaded again.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (are_progress_bars_disabled, disable_progress_bars,
                                       enable_progress_bars)

    root = Path(raw_dir) / "fsd50k"
    include, exclude = set(include), set(exclude)
    wanted = []
    for gt_name, audio_dir in [("dev.csv", FSD50K_HF_LAYOUT["dev_audio_dir"]),
                               ("eval.csv", FSD50K_HF_LAYOUT["eval_audio_dir"])]:
        gt_path = hf_hub_download(FSD50K_HF_REPO, f"{FSD50K_HF_LAYOUT['ground_truth_dir']}/{gt_name}",
                                  repo_type="dataset", local_dir=root)
        gt = pd.read_csv(gt_path)
        selected = gt[[_is_selected(labels, include, exclude) for labels in gt["labels"]]]
        if limit is not None:
            selected = selected.head(limit)
        wanted += [f"{audio_dir}/{fname}.wav" for fname in selected["fname"]]

    def fetch(filename):
        if not (root / filename).exists():
            hf_hub_download(FSD50K_HF_REPO, filename, repo_type="dataset", local_dir=root)

    # One progress line for all clips instead of a Hugging Face bar per clip.
    bars_were_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool,                 Progress("FSD50K", len(wanted), "clips") as progress:
            for future in as_completed([pool.submit(fetch, f) for f in wanted]):
                future.result()
                progress.update()
    finally:
        if not bars_were_disabled:
            enable_progress_bars()
    return select_fsd50k(root, include, exclude, **FSD50K_HF_LAYOUT)


def download_musan_speech(raw_dir, subsets=("librivox",), url=MUSAN_URL, limit=None):
    """Extract MUSAN ``speech/<subset>`` files from the OpenSLR archive; return the MUSAN root.

    The archive (11 GB) is streamed and only the wanted files are written; streaming stops
    as soon as all of them are found. The expected file list comes from MUSAN's
    ``ANNOTATIONS``, fetched from a small Hugging Face mirror. ``limit`` stops after that
    many files per subset (dry runs). Run again to resume: files on disk are skipped.

    Progress lines report the amount downloaded, the speed, the folder of the archive being
    read and the voices found so far. The time left is an upper bound: streaming stops once
    the voices are found, before the end of the archive.
    """
    import tarfile
    import urllib.request

    from huggingface_hub import hf_hub_download

    root = Path(raw_dir) / "musan"
    missing = {}
    for subset in subsets:
        annotations = hf_hub_download(MUSAN_ANNOTATIONS_REPO, f"speech/{subset}/ANNOTATIONS",
                                      repo_type="dataset", local_dir=root)
        names = [line.split()[0] for line in Path(annotations).read_text(encoding="utf-8")
                 .splitlines() if line.strip()]
        if limit is not None:
            names = names[:limit]
        for name in names:
            rel = f"speech/{subset}/{name}.wav"
            if not (root / rel).exists():
                missing[f"musan/{rel}"] = root / rel
    if not missing:
        return root

    wanted = len(missing)
    current = {"folder": "-"}

    def status():
        return f"reading {current['folder']}, voices {wanted - len(missing)}/{wanted}"

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        print(f"MUSAN: streaming {url} ({total / 1e9:.2f} GB) for {wanted} voices", flush=True)
        with Progress("MUSAN", total or None, "GB", scale=1e9, decimals=2, rate_unit="MB/s",
                      rate_scale=1e6, status=status, eta_prefix="at most ") as progress:
            with tarfile.open(fileobj=ProgressFile(response, progress), mode="r|gz") as tar:
                for member in tar:
                    current["folder"] = "/".join(member.name.split("/")[1:3]) or member.name
                    target = missing.pop(member.name, None)
                    if target is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix(".part")
                    with tar.extractfile(member) as src, open(tmp, "wb") as dst:
                        dst.write(src.read())
                    tmp.replace(target)
                    if not missing:
                        break
    if missing:
        raise RuntimeError(f"{len(missing)} MUSAN files were not found in the archive, "
                           f"e.g. {sorted(missing)[:3]}")
    return root


def materialize_pool(pool, data_root, subdir):
    """Copy the pool's files into ``data_root/noise/<subdir>/`` as 16 kHz PCM_16 WAV.

    Returns the pool with ``path`` relative to ``data_root``, so the shared dataset is
    self-contained: whoever attaches it needs neither FSD50K nor MUSAN. Converting traffic
    clips (44.1 kHz) once also saves resampling at every training step. Existing files are
    not rewritten.
    """
    data_root = Path(data_root)
    rows = []
    with Progress(f"noise {subdir}", len(pool), "files") as progress:
        for row in pool.to_dict("records"):
            rows.append(_materialize_row(row, data_root, subdir))
            progress.update()
    return pd.DataFrame(rows, columns=list(pool.columns))


def _materialize_row(row, data_root, subdir):
    rel = Path("noise") / subdir / f"{row['noise_id']}.wav"
    out = data_root / rel
    if not out.exists():
        audio = load_audio(row["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
    return {**row, "path": rel.as_posix(), "duration_s": round(_duration_s(out), 3)}


def take_segment(noise, offset_s, n_samples, sr=SAMPLE_RATE):
    """``n_samples`` of ``noise`` starting at ``offset_s``, looping the noise if it runs out."""
    noise = np.asarray(noise)
    if len(noise) == 0:
        raise ValueError("noise is empty")
    start = int(round(offset_s * sr)) % len(noise)
    reps = int(np.ceil((start + n_samples) / len(noise)))
    return np.tile(noise, reps)[start:start + n_samples]


def read_segment(path, offset_s, n_samples, sr=SAMPLE_RATE):
    """Read only ``n_samples`` (at ``sr``) of a file from ``offset_s``, looping if it runs out.

    Same result as ``take_segment(load_audio(path), offset_s, n_samples)`` for a file
    already at ``sr``, but memory and time depend on the segment, not on the file length.
    Used for long files (MUSAN voices last minutes, babble needs a few seconds).
    """
    info = sf.info(path)
    if info.frames == 0:
        raise ValueError(f"{path} is empty")
    file_sr = info.samplerate
    needed = int(np.ceil(n_samples * file_sr / sr))
    pos = int(round(offset_s * file_sr)) % info.frames
    pieces = []
    with sf.SoundFile(path) as f:
        while needed > 0:
            f.seek(pos)
            piece = f.read(min(needed, info.frames - pos), dtype="float32", always_2d=True)
            pieces.append(piece)
            needed -= len(piece)
            pos = 0
    audio = convert(np.concatenate(pieces), file_sr, sr)
    return np.pad(audio, (0, max(0, n_samples - len(audio))))[:n_samples]


def random_offset_s(rng, noise_duration_s, speech_duration_s):
    """Offset into a noise file: within the part that fits the speech, or anywhere if it must loop."""
    room = noise_duration_s - speech_duration_s
    upper = room if room > 0 else noise_duration_s
    return round(float(rng.uniform(0, upper)), 3)


def band_limit(audio, band_sr, sr=SAMPLE_RATE):
    """Remove content above ``band_sr / 2`` by resampling down to ``band_sr`` and back.

    Used for VietMed: its speech was recorded at 8 kHz, so noise is limited to the same
    band, as if both had passed through the same telephone channel.
    """
    if band_sr >= sr:
        return np.asarray(audio, dtype=np.float32)
    down = convert(audio, sr, band_sr)
    up = convert(down, band_sr, sr)
    n = len(audio)
    return np.pad(up, (0, max(0, n - len(up))))[:n]

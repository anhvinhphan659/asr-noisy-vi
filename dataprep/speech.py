"""Download the speech datasets and turn them into 16 kHz WAV files plus manifests."""

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core import CLIP_THRESHOLD, SAMPLE_RATE, convert, load_audio

from .progress import Progress, ProgressFile
from .text import has_digit, normalize_text

MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0

FLEURS_SPLITS = {"train": "train", "validation": "dev", "test": "test"}
FLEURS_GENDERS = {0: "male", 1: "female", 2: "other"}
VIETMED_SPLITS = ["train", "dev", "test"]


def process_utterance(audio, orig_sr, data_root, dataset, split, utt_id, text_raw,
                      speaker=None, gender=None, **extra):
    """Filter one 16 kHz utterance, write it as PCM_16 if kept, and return its manifest row.

    The row's ``drop_reason`` is None when the utterance is kept, otherwise one of
    ``empty_audio``, ``empty_text``, ``too_short``, ``too_long``. Existing files are
    not rewritten, so a preparation run can be resumed.
    """
    text = normalize_text(text_raw)
    duration_s = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

    if peak == 0.0:
        reason = "empty_audio"
    elif not text:
        reason = "empty_text"
    elif duration_s < MIN_DURATION_S:
        reason = "too_short"
    elif duration_s > MAX_DURATION_S:
        reason = "too_long"
    else:
        reason = None

    rel_path = Path("clean") / dataset / split / f"{utt_id}.wav"
    if reason is None:
        out = Path(data_root) / rel_path
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")

    return {
        "utt_id": utt_id,
        "dataset": dataset,
        "split": split,
        "path": rel_path.as_posix(),
        "duration_s": round(duration_s, 3),
        "orig_sr": int(orig_sr),
        "speaker": speaker,
        "gender": gender,
        "text_raw": text_raw,
        "text": text,
        "has_digit": has_digit(text_raw),
        "peak": round(peak, 5),
        "has_clipping": peak >= CLIP_THRESHOLD,
        **extra,
        "drop_reason": reason,
    }


def save_manifests(rows, manifest_dir, dataset):
    """Write ``clean_<dataset>.csv`` (kept) and ``dropped_<dataset>.csv``; return the kept rows."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    kept = df[df["drop_reason"].isna()].drop(columns="drop_reason").reset_index(drop=True)
    dropped = df[df["drop_reason"].notna()].reset_index(drop=True)
    kept.to_csv(manifest_dir / f"clean_{dataset}.csv", index=False)
    dropped.to_csv(manifest_dir / f"dropped_{dataset}.csv", index=False)
    return kept


def decode_bytes(data):
    """Decode an encoded audio file (WAV, FLAC, OGG...) to 16 kHz mono; return (audio, orig_sr)."""
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return convert(x, sr), sr


def iter_parquet(path, columns, limit=None, batch_size=16):
    """Stream rows of a parquet file as dicts without loading the whole file."""
    import pyarrow.parquet as pq

    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            if limit is not None and n >= limit:
                return
            n += 1
            yield row


def _parquet_rows(path, limit=None):
    import pyarrow.parquet as pq

    n = pq.ParquetFile(path).metadata.num_rows
    return n if limit is None else min(n, limit)


def _kept_status(rows, start):
    """Status for a progress line: utterances kept among ``rows[start:]``."""
    kept = sum(r["drop_reason"] is None for r in rows[start:])
    return f"kept {kept}/{len(rows) - start}"


def _raw(data_root, raw_dir):
    """Where downloads go: ``raw_dir`` if given (keep it outside the shared dataset), else
    ``data_root/raw``."""
    return Path(raw_dir) if raw_dir is not None else Path(data_root) / "raw"


def _download(repo_id, filename, local_dir, revision=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision,
                           local_dir=local_dir)


# --- FLEURS -------------------------------------------------------------------------------

def prepare_fleurs(data_root, limit=None, raw_dir=None):
    """FLEURS ``vi_vn``: official splits, ``validation`` renamed ``dev``. Returns manifest rows.

    FLEURS has no speaker IDs; ``sentence_id`` identifies the read sentence (several
    speakers can read the same one), and ``utt_id`` comes from the unique audio file name.
    """
    data_root = Path(data_root)
    rows = []
    for hf_split, split in FLEURS_SPLITS.items():
        parquet = _download("google/fleurs", f"vi_vn/{hf_split}/0000.parquet",
                            _raw(data_root, raw_dir) / "fleurs", revision="refs/convert/parquet")
        columns = ["id", "audio", "raw_transcription", "gender"]
        start = len(rows)
        with Progress(f"FLEURS {split}", _parquet_rows(parquet, limit), "utterances",
                      status=lambda start=start: _kept_status(rows, start)) as progress:
            for row in iter_parquet(parquet, columns, limit):
                audio, orig_sr = decode_bytes(row["audio"]["bytes"])
                rows.append(process_utterance(
                    audio, orig_sr, data_root, "fleurs", split,
                    "fleurs_" + Path(row["audio"]["path"]).stem, row["raw_transcription"],
                    gender=FLEURS_GENDERS.get(row["gender"], row["gender"]),
                    sentence_id=row["id"],
                ))
                progress.update()
    return rows


# --- VIVOS --------------------------------------------------------------------------------

def read_vivos_split(split_dir):
    """Read one VIVOS split folder; return ``[(utt_id, speaker, text)]`` and ``{speaker: gender}``."""
    split_dir = Path(split_dir)
    items = []
    with open(split_dir / "prompts.txt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                utt_id, text = line.strip().split(" ", 1)
                items.append((utt_id, utt_id.split("_")[0], text))
    genders = {}
    gender_file = split_dir / "genders.txt"
    if gender_file.exists():
        with open(gender_file, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    genders[parts[0]] = {"m": "male", "f": "female"}.get(parts[1].lower(), parts[1])
    return items, genders


def choose_dev_speakers(speakers, genders, per_gender, seed):
    """Pick ``per_gender`` speakers of each gender, reproducibly, to hold out as dev."""
    rng = np.random.default_rng(seed)
    speakers = sorted(set(speakers))
    chosen = set()
    for gender in sorted({genders.get(s) for s in speakers}, key=str):
        pool = [s for s in speakers if genders.get(s) == gender]
        k = min(per_gender, len(pool))
        chosen.update(rng.choice(pool, size=k, replace=False).tolist())
    return chosen


def prepare_vivos(data_root, dev_speakers_per_gender=2, seed=42, limit=None, raw_dir=None):
    """VIVOS: train / test kept, dev carved out of train by speaker (balanced by gender).

    Returns ``(rows, dev_speakers)``.
    """
    data_root = Path(data_root)
    raw = _raw(data_root, raw_dir) / "vivos"
    tar_path = _download("AILAB-VNUHCM/vivos", "data/vivos.tar.gz", raw)
    root = raw / "vivos"
    done_marker = raw / ".extracted"
    if not done_marker.exists():  # a stopped extraction is redone from the start
        size = Path(tar_path).stat().st_size
        with open(tar_path, "rb") as f, Progress("VIVOS extract", size, "GB", scale=1e9,
                                                 decimals=2, rate_unit="MB/s",
                                                 rate_scale=1e6) as progress:
            with tarfile.open(fileobj=ProgressFile(f, progress), mode="r|gz") as tar:
                try:
                    tar.extractall(raw, filter="data")
                except TypeError:  # Python without extraction filters
                    tar.extractall(raw)
        done_marker.touch()

    train_items, train_genders = read_vivos_split(root / "train")
    test_items, test_genders = read_vivos_split(root / "test")
    dev_speakers = choose_dev_speakers([s for _, s, _ in train_items], train_genders,
                                       dev_speakers_per_gender, seed)

    jobs = [("dev" if spk in dev_speakers else "train", "train", utt, spk, text, train_genders)
            for utt, spk, text in train_items]
    jobs += [("test", "test", utt, spk, text, test_genders) for utt, spk, text in test_items]

    if limit is not None:
        counts, kept_jobs = {}, []
        for job in jobs:
            counts[job[0]] = counts.get(job[0], 0) + 1
            if counts[job[0]] <= limit:
                kept_jobs.append(job)
        jobs = kept_jobs

    rows = []
    with Progress("VIVOS", len(jobs), "utterances",
                  status=lambda: _kept_status(rows, 0)) as progress:
        for split, src_split, utt, spk, text, genders in jobs:
            wav = root / src_split / "waves" / spk / f"{utt}.wav"
            rows.append(process_utterance(
                load_audio(wav), sf.info(wav).samplerate, data_root, "vivos", split,
                f"vivos_{utt}", text, speaker=spk, gender=genders.get(spk),
            ))
            progress.update()
    return rows, dev_speakers


# --- VietMed ------------------------------------------------------------------------------

def prepare_vietmed(data_root, limit=None, raw_dir=None):
    """VietMed: official train / dev / test. Audio is 8 kHz OGG, upsampled to 16 kHz.

    ``rec_condition`` (Tel, Consultation, News...) is kept for analysis.
    """
    data_root = Path(data_root)
    rows = []
    for split in VIETMED_SPLITS:
        parquet = _download("leduckhai/VietMed", f"data/{split}-00000-of-00001.parquet",
                            _raw(data_root, raw_dir) / "vietmed")
        columns = ["audio", "text", "utterance_id", "speaker_name", "gender", "rec_condition"]
        start = len(rows)
        with Progress(f"VietMed {split}", _parquet_rows(parquet, limit), "utterances",
                      status=lambda start=start: _kept_status(rows, start)) as progress:
            for row in iter_parquet(parquet, columns, limit):
                audio, orig_sr = decode_bytes(row["audio"]["bytes"])
                rows.append(process_utterance(
                    audio, orig_sr, data_root, "vietmed", split,
                    f"vietmed_{row['utterance_id']}", row["text"],
                    speaker=row["speaker_name"],
                    gender=row["gender"].lower() if row["gender"] else None,
                    rec_condition=row["rec_condition"],
                ))
                progress.update()
    return rows


# --- Manifest utilities -------------------------------------------------------------------

def subsample(manifest, n, by=None, seed=42):
    """Return ``n`` rows of ``manifest``, reproducibly, keeping the proportions of column ``by``.

    Used to shrink a large test set (e.g. VietMed test, stratified by ``rec_condition``).
    """
    if n >= len(manifest):
        return manifest.copy()
    if by is None:
        return manifest.sample(n=n, random_state=seed).sort_index()
    shares = manifest[by].value_counts(normalize=True)
    quotas = (shares * n).round().astype(int)
    quotas.iloc[0] += n - quotas.sum()  # make the quotas add up to n exactly
    parts = [manifest[manifest[by] == value].sample(n=min(k, (manifest[by] == value).sum()),
                                                    random_state=seed)
             for value, k in quotas.items()]
    return pd.concat(parts).sort_index()


def speaker_overlap(manifest, key="speaker"):
    """Count values of ``key`` shared between splits of each dataset (leakage check)."""
    rows = []
    for dataset, group in manifest.groupby("dataset"):
        sets = {s: set(g[key].dropna()) for s, g in group.groupby("split")}
        for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            if a in sets and b in sets:
                rows.append({"dataset": dataset, "key": key, "splits": f"{a}/{b}",
                             "shared": len(sets[a] & sets[b])})
    return pd.DataFrame(rows)


def summarize_manifest(manifest):
    """Utterances, hours and basic quality figures per dataset and split."""
    return manifest.groupby(["dataset", "split"], sort=False).agg(
        utterances=("utt_id", "size"),
        hours=("duration_s", lambda d: round(d.sum() / 3600, 2)),
        mean_s=("duration_s", lambda d: round(d.mean(), 2)),
        max_s=("duration_s", "max"),
        speakers=("speaker", "nunique"),
        orig_sr=("orig_sr", lambda s: ",".join(map(str, sorted(s.unique())))),
        pct_digit=("has_digit", lambda s: round(100 * s.mean(), 1)),
        clipped=("has_clipping", "sum"),
    )

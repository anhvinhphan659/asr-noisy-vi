"""Ready-to-use data for the model notebooks: evaluation batches and the training dataset.

Everything here is model-independent: audio comes out as 16 kHz float32 arrays and text as
normalized transcripts. Feature extraction and tokenization stay in the model notebooks.
"""

from pathlib import Path

import pandas as pd

from core import load_audio

from .render import NoiseBank, render

EVAL_COLUMNS = ["utt_id", "condition", "dataset", "noise_type", "snr_db", "path", "text",
                "duration_s"]


def _read(table):
    return table.copy() if isinstance(table, pd.DataFrame) else pd.read_csv(table)


def load_eval_set(plan, manifest, datasets=None, conditions=None, limit=None, seed=42):
    """Rows to transcribe for a test or dev plan, with their reference text and duration.

    ``plan`` and ``manifest`` are DataFrames or CSV paths (``test_plan.csv`` /
    ``dev_plan.csv`` and ``clean.csv``). ``datasets`` / ``conditions`` keep only some of them;
    ``limit`` keeps that many utterances per dataset (all their conditions), for a dry run.
    """
    plan, manifest = _read(plan), _read(manifest)
    if datasets is not None:
        plan = plan[plan["dataset"].isin(datasets)]
    if conditions is not None:
        plan = plan[plan["condition"].isin(conditions)]
    if limit is not None:
        utts = plan[["dataset", "utt_id"]].drop_duplicates()
        keep = set()
        for _, group in utts.groupby("dataset"):
            keep.update(group["utt_id"].sample(n=min(limit, len(group)), random_state=seed))
        plan = plan[plan["utt_id"].isin(keep)]
    refs = manifest[["utt_id", "text", "duration_s"]]
    out = plan.merge(refs, on="utt_id", how="left", validate="many_to_one")
    if out["text"].isna().any():
        raise ValueError("some plan utterances are missing from the manifest")
    return out[EVAL_COLUMNS].reset_index(drop=True)


def iter_eval_batches(eval_set, data_root, batch_size=16, sort_by_duration=True):
    """Yield ``(rows, audios)``: a DataFrame slice and its 16 kHz audio arrays.

    Sorting by duration puts utterances of similar length together, which reduces padding
    and speeds up batched inference for both models. The order does not affect results:
    predictions are matched by ``utt_id`` and ``condition``.
    """
    data_root = Path(data_root)
    if sort_by_duration:
        eval_set = eval_set.sort_values("duration_s", kind="stable")
    for start in range(0, len(eval_set), batch_size):
        rows = eval_set.iloc[start:start + batch_size]
        yield rows, [load_audio(data_root / p) for p in rows["path"]]


class TrainSet:
    """Training utterances as ``{"utt_id", "audio", "text", "noise_type", "snr_db"}``.

    Without ``schedule`` it serves clean audio (variant ``ft-clean``). With a training
    schedule and a ``NoiseBank`` it renders each sample with the schedule row of the
    current epoch (variant ``ft-noisy``), so every model sees the same noisy samples.

    It has ``__len__`` and ``__getitem__`` and can be passed to ``torch.utils.data.DataLoader``
    directly. Call ``set_epoch`` before iterating each epoch. With ``persistent_workers=True``
    the workers keep their own copy and do not see the new epoch: leave it False.
    """

    def __init__(self, manifest, data_root, schedule=None, bank=None, split="train",
                 datasets=None):
        manifest = _read(manifest)
        manifest = manifest[manifest["split"] == split]
        if datasets is not None:
            manifest = manifest[manifest["dataset"].isin(datasets)]
        if manifest.empty:
            raise ValueError("no utterances selected")
        self.data_root = Path(data_root)
        self.utt_ids = manifest["utt_id"].tolist()
        self.paths = manifest["path"].tolist()
        self.texts = manifest["text"].tolist()
        self.epoch = 1
        self.specs = None
        if schedule is not None:
            if bank is None:
                raise ValueError("a schedule needs a NoiseBank")
            schedule = _read(schedule)
            schedule = schedule[schedule["utt_id"].isin(set(self.utt_ids))]
            self.specs = {(int(r["epoch"]), r["utt_id"]): r
                          for r in schedule.to_dict("records")}
            self.epochs = sorted({epoch for epoch, _ in self.specs})
            for epoch in self.epochs:
                missing = [u for u in self.utt_ids if (epoch, u) not in self.specs]
                if missing:
                    raise ValueError(f"schedule epoch {epoch} lacks {len(missing)} utterances, "
                                     f"e.g. {missing[:3]}")
        self.bank = bank

    def set_epoch(self, epoch):
        if self.specs is not None and epoch not in self.epochs:
            raise ValueError(f"the schedule has epochs {self.epochs}, not {epoch}")
        self.epoch = epoch

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, i):
        utt_id = self.utt_ids[i]
        audio = load_audio(self.data_root / self.paths[i])
        noise_type, snr_db = "clean", None
        if self.specs is not None:
            spec = self.specs[(self.epoch, utt_id)]
            audio = render(audio, spec, self.bank)
            noise_type, snr_db = spec["noise_type"], spec["snr_db"]
        return {"utt_id": utt_id, "audio": audio, "text": self.texts[i],
                "noise_type": noise_type, "snr_db": snr_db}


def collate_raw(items):
    """Stack ``TrainSet`` items into lists, ready for a model's processor / tokenizer."""
    return {key: [item[key] for item in items] for key in items[0]}


def load_train_set(manifest, data_root, variant, schedule=None, noise_pools=(), datasets=None):
    """``TrainSet`` for a variant name: ``"ft-clean"`` or ``"ft-noisy"``.

    ``noise_pools`` are the pool DataFrames or CSV paths (traffic, voices) needed by
    ``"ft-noisy"``.
    """
    if variant == "ft-clean":
        return TrainSet(manifest, data_root, datasets=datasets)
    if variant == "ft-noisy":
        if schedule is None or not noise_pools:
            raise ValueError("ft-noisy needs a schedule and the noise pools")
        bank = NoiseBank(*[_read(p) for p in noise_pools], data_root=data_root)
        return TrainSet(manifest, data_root, schedule=schedule, bank=bank, datasets=datasets)
    raise ValueError(f"unknown variant {variant!r}")

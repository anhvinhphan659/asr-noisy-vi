"""Write model outputs in the prediction format, resuming an interrupted run."""

import csv
from pathlib import Path

import pandas as pd

from dataprep.progress import Progress

from .scoring import PREDICTION_COLUMNS


class PredictionWriter:
    """Append predictions of one system to a CSV, skipping rows already written.

    A Kaggle session can stop before a run ends; running the same notebook again resumes
    where it stopped::

        with PredictionWriter("predictions_qwen3-asr-0.6b_zero-shot.csv",
                              "qwen3-asr-0.6b", "zero-shot") as writer:
            for rows, audios in iter_eval_batches(writer.pending(eval_set), DATA_ROOT):
                writer.add(rows, transcribe(audios))

    Rows are flushed to disk after every ``add``. After ``pending``, progress lines count
    the rows written against the rows that were pending.
    """

    def __init__(self, path, model, variant):
        self.path = Path(path)
        self.model, self.variant = model, variant
        self.done = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            existing = pd.read_csv(self.path, dtype=str, keep_default_na=False)
            other = existing[(existing["model"] != model) | (existing["variant"] != variant)]
            if not other.empty:
                raise ValueError(f"{self.path} holds predictions of another system: "
                                 f"{other[['model', 'variant']].drop_duplicates().values.tolist()}")
            self.done = set(zip(existing["utt_id"], existing["condition"]))
        self._file = None
        self._writer = None
        self._progress = None

    def __enter__(self):
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if new:
            self._writer.writerow(PREDICTION_COLUMNS)
        return self

    def __exit__(self, *exc):
        if self._progress is not None:
            self._progress.close()
        self._file.close()
        self._file = self._writer = None

    def pending(self, eval_set):
        """The rows of ``eval_set`` that have no prediction yet."""
        keys = list(zip(eval_set["utt_id"], eval_set["condition"]))
        mask = [key not in self.done for key in keys]
        pending = eval_set[mask]
        already = len(keys) - len(pending)
        self._progress = Progress(f"predictions {self.model} {self.variant}", len(pending),
                                  "rows", status=f"{already} already written before")
        return pending

    def add(self, rows, hyps):
        """Write one hypothesis per row of ``rows`` (a DataFrame slice), in the same order."""
        hyps = list(hyps)
        if len(hyps) != len(rows):
            raise ValueError(f"{len(rows)} rows but {len(hyps)} hypotheses")
        if self._writer is None:
            raise RuntimeError("use PredictionWriter inside a 'with' block")
        written = 0
        for utt_id, condition, hyp in zip(rows["utt_id"], rows["condition"], hyps):
            if (utt_id, condition) in self.done:
                continue
            self._writer.writerow([utt_id, condition, self.model, self.variant,
                                   " ".join(str(hyp).split())])
            self.done.add((utt_id, condition))
            written += 1
        self._file.flush()
        if self._progress is not None:
            self._progress.update(written)

"""Progress lines for long steps: the same one-line format everywhere, about once a minute.

    FLEURS train: 1250/2994 utterances (41.8%), 21.3/s, ~2 min left, kept 1247

Long loops create a ``Progress`` and call ``update``; reading a large download or archive
goes through ``ProgressFile``. ``set_progress_interval`` changes how often lines are
printed for the whole session (``None`` turns them off).
"""

import time

_settings = {"interval_s": 60.0}
_DEFAULT = object()


def set_progress_interval(seconds):
    """Seconds between progress lines for every step; ``None`` disables progress output."""
    _settings["interval_s"] = None if seconds is None else float(seconds)


def get_progress_interval():
    return _settings["interval_s"]


class Progress:
    """Counts work done and prints a line every ``interval_s`` and once when closed.

    ``total`` may be unknown (None). ``scale`` and ``rate_scale`` turn raw counts into the
    printed units, e.g. bytes shown as GB with a rate in MB/s. ``status`` is a string or a
    function returning one, appended to each line.
    """

    def __init__(self, label, total=None, unit="items", interval_s=_DEFAULT, scale=1.0,
                 decimals=0, rate_unit=None, rate_scale=None, status=None,
                 eta_prefix="~"):
        self.label, self.total, self.unit = label, total, unit
        self.interval_s = get_progress_interval() if interval_s is _DEFAULT else interval_s
        self.scale, self.decimals = scale, decimals
        self.rate_unit = rate_unit or f"{unit}/s"
        self.rate_scale = rate_scale or scale
        self.status, self.eta_prefix = status, eta_prefix
        self.done = 0
        self.start = self.last = time.monotonic()
        self.last_done = 0
        self.closed = False

    @property
    def enabled(self):
        return self.interval_s is not None

    def update(self, n=1, status=None):
        self.done += n
        if status is not None:
            self.status = status
        if self.enabled:
            now = time.monotonic()
            if now - self.last >= self.interval_s:
                self.report(now)

    def _amount(self, value):
        return f"{value / self.scale:.{self.decimals}f}"

    def line(self, now=None, final=False):
        now = time.monotonic() if now is None else now
        parts = [f"{self.label}: {self._amount(self.done)}"]
        if self.total:
            parts[0] += f"/{self._amount(self.total)} {self.unit} ({100 * self.done / self.total:.1f}%)"
        else:
            parts[0] += f" {self.unit}"
        if final:
            parts.append(f"done in {(now - self.start) / 60:.1f} min")
        else:
            window = max(now - self.last, 1e-9)
            rate = (self.done - self.last_done) / window / self.rate_scale
            parts.append(f"{rate:.1f} {self.rate_unit}")
            mean = self.done / max(now - self.start, 1e-9)
            if self.total and mean > 0:
                left_min = (self.total - self.done) / mean / 60
                parts.append(f"{self.eta_prefix}{left_min:.0f} min left")
        status = self.status() if callable(self.status) else self.status
        if status:
            parts.append(str(status))
        return ", ".join(parts)

    def report(self, now=None):
        now = time.monotonic() if now is None else now
        print(self.line(now), flush=True)
        self.last, self.last_done = now, self.done

    def close(self):
        if self.enabled and not self.closed:
            print(self.line(final=True), flush=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ProgressFile:
    """A read-only file wrapper that reports the bytes read to a ``Progress``."""

    def __init__(self, raw, progress):
        self.raw, self.progress = raw, progress

    def read(self, n=-1):
        data = self.raw.read(n)
        self.progress.update(len(data))
        return data

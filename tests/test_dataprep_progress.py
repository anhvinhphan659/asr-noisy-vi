import io

import pytest

from dataprep import Progress, ProgressFile, get_progress_interval, set_progress_interval


@pytest.fixture
def every_call():
    """Print a progress line at every update, restore the default afterwards."""
    before = get_progress_interval()
    set_progress_interval(0)
    yield
    set_progress_interval(before)


def test_progress_line_format(every_call, capsys):
    with Progress("FLEURS train", 4, "utterances", status=lambda: "kept 2") as progress:
        progress.update()
        progress.update()
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("FLEURS train: 1/4 utterances (25.0%), ")
    assert "utterances/s" in lines[0] and "min left" in lines[0] and lines[0].endswith("kept 2")
    assert lines[-1].startswith("FLEURS train: 2/4 utterances (50.0%), done in ")


def test_progress_without_total(every_call, capsys):
    with Progress("x", None, "files") as progress:
        progress.update(3)
    assert capsys.readouterr().out.startswith("x: 3 files, ")


def test_progress_bytes_as_gb(every_call, capsys):
    progress = Progress("MUSAN", 4e9, "GB", scale=1e9, decimals=2, rate_unit="MB/s",
                        rate_scale=1e6, eta_prefix="at most ")
    reader = ProgressFile(io.BytesIO(b"x" * 1000), progress)
    assert reader.read(500) == b"x" * 500
    line = capsys.readouterr().out
    assert line.startswith("MUSAN: 0.00/4.00 GB (0.0%), ") and "MB/s" in line
    assert "at most" in line


def test_progress_interval_limits_output(capsys):
    before = get_progress_interval()
    set_progress_interval(3600)
    try:
        with Progress("slow", 100, "files") as progress:
            for _ in range(100):
                progress.update()
    finally:
        set_progress_interval(before)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "done in" in lines[0]  # only the closing line


def test_progress_can_be_disabled(capsys):
    before = get_progress_interval()
    set_progress_interval(None)
    try:
        with Progress("quiet", 10) as progress:
            progress.update(10)
    finally:
        set_progress_interval(before)
    assert capsys.readouterr().out == ""

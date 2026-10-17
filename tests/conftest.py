import numpy as np
import pytest

from core import SAMPLE_RATE


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def speech():
    """2 s speech-like signal: a modulated tone with a silent gap."""
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2
    x[SAMPLE_RATE // 2 : SAMPLE_RATE // 2 + 2000] = 0.0
    return x.astype(np.float32)


@pytest.fixture
def noise(rng):
    """3 s of white noise."""
    return (0.1 * rng.standard_normal(3 * SAMPLE_RATE)).astype(np.float32)

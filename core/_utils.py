import numpy as np


def as_audio(x, name="audio", max_ndim=1):
    """Return ``x`` as a float array with at most ``max_ndim`` dimensions.

    Lists are converted; integer arrays are rejected because their value
    scale (e.g. int16 PCM) is ambiguous.
    """
    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError(f"{name} must be floating point, got {x.dtype}")
    else:
        x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.ndim > max_ndim:
        raise ValueError(f"{name} must have 1 to {max_ndim} dimensions, got shape {x.shape}")
    return x


def power(x):
    """Mean of squares in float64; 0.0 for an empty array."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x**2)) if x.size else 0.0

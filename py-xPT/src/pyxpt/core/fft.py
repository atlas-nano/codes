"""Shared windowing and FFT utilities.

The FFT helpers (``rfft``, ``irfft``, ``fft``, ``ifft``, ``rfftfreq``,
``fftfreq``) are signature-compatible drop-in replacements for the
corresponding ``numpy.fft`` calls.  Internally they dispatch to
``scipy.fft`` (multi-threaded pocketfft) with ``workers`` set from the
``PYXPT_FFT_WORKERS`` environment variable (default ``-1`` = use all
available cores).  This typically yields a 4-6× speed-up on the large
1-D transforms that dominate the VACF, ACF, and DoS computations on
multicore machines, with no API change at the call sites.

If ``scipy.fft`` is not available the helpers fall back transparently to
``numpy.fft``.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# ── Multi-threaded FFT dispatch ────────────────────────────────────────────
try:
    from scipy import fft as _scipy_fft
    _HAVE_SCIPY_FFT = True
except ImportError:                                                  # pragma: no cover
    _scipy_fft = None
    _HAVE_SCIPY_FFT = False
    log.warning(
        "scipy.fft unavailable; falling back to single-threaded numpy.fft. "
        "Install scipy ≥ 1.10 for multi-threaded FFTs."
    )


def _resolve_workers(workers: int | None) -> int:
    """Return the worker count for a scipy.fft call.

    If ``workers`` is None (caller didn't override), use the value from
    ``PYXPT_FFT_WORKERS`` (default ``-1`` = all cores).
    """
    if workers is not None:
        return workers
    return int(os.environ.get("PYXPT_FFT_WORKERS", "-1"))


def rfft(a, n=None, axis=-1, norm=None, workers=None):
    """Real-input 1-D FFT.  Drop-in replacement for ``np.fft.rfft``."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.rfft(a, n=n, axis=axis, norm=norm,
                                workers=_resolve_workers(workers))
    return np.fft.rfft(a, n=n, axis=axis, norm=norm)


def irfft(a, n=None, axis=-1, norm=None, workers=None):
    """Inverse of :func:`rfft`.  Drop-in for ``np.fft.irfft``."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.irfft(a, n=n, axis=axis, norm=norm,
                                 workers=_resolve_workers(workers))
    return np.fft.irfft(a, n=n, axis=axis, norm=norm)


def fft(a, n=None, axis=-1, norm=None, workers=None):
    """Complex-input 1-D FFT.  Drop-in for ``np.fft.fft``."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.fft(a, n=n, axis=axis, norm=norm,
                               workers=_resolve_workers(workers))
    return np.fft.fft(a, n=n, axis=axis, norm=norm)


def ifft(a, n=None, axis=-1, norm=None, workers=None):
    """Inverse of :func:`fft`.  Drop-in for ``np.fft.ifft``."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.ifft(a, n=n, axis=axis, norm=norm,
                                workers=_resolve_workers(workers))
    return np.fft.ifft(a, n=n, axis=axis, norm=norm)


def rfftfreq(n, d=1.0):
    """Frequency axis for :func:`rfft` output."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.rfftfreq(n, d=d)
    return np.fft.rfftfreq(n, d=d)


def fftfreq(n, d=1.0):
    """Frequency axis for :func:`fft` output."""
    if _HAVE_SCIPY_FFT:
        return _scipy_fft.fftfreq(n, d=d)
    return np.fft.fftfreq(n, d=d)


def configured_workers() -> int:
    """Diagnostic: return the worker count that helpers use by default."""
    return _resolve_workers(None)


def fft_backend_name() -> str:
    """Diagnostic: ``'scipy'`` or ``'numpy'``."""
    return "scipy" if _HAVE_SCIPY_FFT else "numpy"


def make_lag_window(M: int, apodization: str, apodization_alpha: float = 0.5) -> np.ndarray:
    """
    One-sided lag window for apodizing a correlation function before the final IFFT.

    Returns a length-M array with w[0]=1 tapering to w[M-1]~0, constructed
    by taking the second half of a symmetric window of length 2M-1.  This
    preserves the zero-lag (most-correlated) value while smoothly suppressing
    large lags, reducing spectral leakage (Gibbs ringing) from the abrupt
    correlation-length truncation.

    Parameters
    ----------
    M : int
        Number of lag points (one-sided length).
    apodization : str
        Window name: ``"none"``, ``"hann"``, ``"hamming"``, ``"blackman"``, or ``"tukey"``.
    apodization_alpha : float
        Taper fraction for the Tukey window (α=0 → rectangular, α=1 → Hann).

    Returns
    -------
    w : np.ndarray of shape (M,)
    """
    name = apodization.lower()
    if name == "none":
        return np.ones(M, dtype=np.float64)
    from scipy.signal import windows as _wins
    if name == "hann":
        full = _wins.hann(2 * M - 1)
    elif name == "hamming":
        full = _wins.hamming(2 * M - 1)
    elif name == "blackman":
        full = _wins.blackman(2 * M - 1)
    elif name == "tukey":
        full = _wins.tukey(2 * M - 1, alpha=apodization_alpha)
    else:
        raise ValueError(
            f"Unknown apodization window {apodization!r}. "
            "Choose: none, hann, hamming, blackman, tukey"
        )
    return full[M - 1:].copy().astype(np.float64)

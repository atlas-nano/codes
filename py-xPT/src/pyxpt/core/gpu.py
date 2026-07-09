# -*- coding: utf-8 -*-
"""
GPU backend selection and array abstractions for py-xPT

Provides a unified interface to use either NumPy (CPU) or CuPy (GPU) depending
on availability and user choice. FFT operations and array manipulations are
abstracted through this module.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from . import fft as _onfft   # multi-threaded scipy.fft wrapper

log = logging.getLogger(__name__)

# GPU backend state
_BACKEND: Literal["numpy", "cupy"] = "numpy"
_GPU_AVAILABLE = False

try:
    import cupy as cp
    _GPU_AVAILABLE = True
    log.info("CuPy is available; GPU acceleration can be enabled with --gpu flag")
except ImportError:  # pragma: no cover
    log.debug("CuPy not installed; GPU acceleration unavailable")


class ArrayBackend:
    """
    Unified array backend (CPU or GPU) for FFT and array operations.
    Gracefully falls back to NumPy if CuPy/CUDA initialization fails.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the array backend.
        
        Parameters
        ----------
        use_gpu : bool
            If True, use CuPy (GPU). If False or CuPy unavailable, use NumPy.
        """
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self._cuda_failed = False  # Track if CUDA fails at runtime
        
        if use_gpu and not _GPU_AVAILABLE:
            log.warning("--gpu requested but CuPy not installed; falling back to NumPy")
        
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            log.info("GPU acceleration enabled (CuPy)")
        else:
            log.debug("Using NumPy (CPU)")
    
    def to_gpu(self, arr: np.ndarray) -> np.ndarray | cp.ndarray:
        """Transfer array to GPU (if enabled and CUDA working)."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.asarray(arr)
            except (ImportError, OSError) as e:
                log.warning(f"CUDA transfer failed: {e}. Using NumPy instead.")
                self._cuda_failed = True
                return arr
        return arr
    
    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array back to CPU."""
        try:
            if self.use_gpu and hasattr(arr, 'get'):
                return cp.asnumpy(arr)
        except (ImportError, OSError, AttributeError):
            pass
        return np.asarray(arr)
    
    def zeros(self, shape, dtype=None):
        """Create zero array on current backend."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.zeros(shape, dtype=dtype)
            except (ImportError, OSError):
                self._cuda_failed = True
        return np.zeros(shape, dtype=dtype)
    
    def zeros_like(self, arr):
        """Create zero array with same shape/dtype as input."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.zeros_like(arr)
            except (ImportError, OSError):
                self._cuda_failed = True
        return np.zeros_like(np.asarray(arr))
    
    def asarray(self, arr, dtype=None):
        """Convert to array on current backend."""
        if self.use_gpu and not self._cuda_failed:
            try:
                if dtype is not None:
                    return cp.asarray(arr, dtype=dtype)
                return cp.asarray(arr)
            except (ImportError, OSError):
                self._cuda_failed = True
        if dtype is not None:
            return np.asarray(arr, dtype=dtype)
        return np.asarray(arr)
    
    def fft(self, arr, axis: int = -1, **kwargs):
        """FFT on current backend. Falls back to NumPy if CUDA unavailable."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.fft.fft(arr, axis=axis, **kwargs)
            except (ImportError, OSError) as e:
                log.warning(
                    f"CUDA library error during FFT: {e}\n"
                    "Falling back to NumPy for remaining computations"
                )
                self._cuda_failed = True
                self.xp = np
                # Convert input to NumPy if needed
                arr_np = cp.asnumpy(arr) if hasattr(arr, 'get') else np.asarray(arr)
                return _onfft.fft(arr_np, axis=axis, **kwargs)
        return _onfft.fft(arr, axis=axis, **kwargs)
    
    def ifft(self, arr, axis: int = -1, **kwargs):
        """Inverse FFT on current backend. Falls back to NumPy if CUDA unavailable."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.fft.ifft(arr, axis=axis, **kwargs)
            except (ImportError, OSError) as e:
                log.warning(
                    f"CUDA library error during IFFT: {e}\n"
                    "Falling back to NumPy for remaining computations"
                )
                self._cuda_failed = True
                self.xp = np
                arr_np = cp.asnumpy(arr) if hasattr(arr, 'get') else np.asarray(arr)
                return _onfft.ifft(arr_np, axis=axis, **kwargs)
        return _onfft.ifft(arr, axis=axis, **kwargs)
    
    def rfft(self, arr, axis: int = -1, **kwargs):
        """Real FFT on current backend. Falls back to NumPy if CUDA unavailable."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.fft.rfft(arr, axis=axis, **kwargs)
            except (ImportError, OSError) as e:
                log.warning(
                    f"CUDA library error during RFFT: {e}\n"
                    "Falling back to NumPy for remaining computations"
                )
                self._cuda_failed = True
                self.xp = np
                arr_np = cp.asnumpy(arr) if hasattr(arr, 'get') else np.asarray(arr)
                return _onfft.rfft(arr_np, axis=axis, **kwargs)
        return _onfft.rfft(arr, axis=axis, **kwargs)
    
    def irfft(self, arr, n: int | None = None, axis: int = -1, **kwargs):
        """Inverse real FFT on current backend. Falls back to NumPy if CUDA unavailable."""
        if self.use_gpu and not self._cuda_failed:
            try:
                return cp.fft.irfft(arr, n=n, axis=axis, **kwargs)
            except (ImportError, OSError) as e:
                log.warning(
                    f"CUDA library error during IRFFT: {e}\n"
                    "Falling back to NumPy for remaining computations"
                )
                self._cuda_failed = True
                self.xp = np
                arr_np = cp.asnumpy(arr) if hasattr(arr, 'get') else np.asarray(arr)
                return _onfft.irfft(arr_np, n=n, axis=axis, **kwargs)
        return _onfft.irfft(arr, n=n, axis=axis, **kwargs)
    
    def sum(self, arr, axis=None, **kwargs):
        """Sum on current backend."""
        return self.xp.sum(arr, axis=axis, **kwargs)
    
    def mean(self, arr, axis=None, **kwargs):
        """Mean on current backend."""
        return self.xp.mean(arr, axis=axis, **kwargs)
    
    def std(self, arr, axis=None, **kwargs):
        """Standard deviation on current backend."""
        return self.xp.std(arr, axis=axis, **kwargs)
    
    def conj(self, arr):
        """Complex conjugate on current backend."""
        return self.xp.conj(arr)
    
    def real(self, arr):
        """Real part on current backend."""
        return self.xp.real(arr)
    
    def abs(self, arr):
        """Absolute value on current backend."""
        return self.xp.abs(arr)
    
    def sqrt(self, arr):
        """Square root on current backend."""
        return self.xp.sqrt(arr)
    
    def exp(self, arr):
        """Exponential on current backend."""
        return self.xp.exp(arr)
    
    def log(self, arr):
        """Natural logarithm on current backend."""
        return self.xp.log(arr)
    
    def dot(self, a, b):
        """Dot product on current backend."""
        return self.xp.dot(a, b)


def get_backend(use_gpu: bool = False) -> ArrayBackend:
    """
    Get an array backend (CPU or GPU).
    
    Parameters
    ----------
    use_gpu : bool
        Request GPU acceleration if available.
    
    Returns
    -------
    ArrayBackend
        Backend instance for array operations.
    """
    return ArrayBackend(use_gpu=use_gpu)


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE

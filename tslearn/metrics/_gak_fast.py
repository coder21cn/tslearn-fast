"""Fused parallel cdist_gak kernel for the numpy backend.

Mirrors `_dtw_fast.py` / `_softdtw_fast.py`: replaces the per-pair Python
wrapper plus per-pair joblib scheduling in `_cdist_generic` with a single
`@njit(parallel=True)` kernel that runs the GAK DP inline with `prange` over
rows.

The GAK gram-matrix correction (Cuturi 2011) is folded into the inner cell
update: ``K(s1[i], s2[j]) = g / (2 - g)`` where
``g = exp(-||s1[i] - s2[j]||^2 / (2*sigma^2))``. This avoids materializing
the full gram matrix per pair.
"""
from numba import njit, prange

import numpy

from .utils import (
    _RowProgress,
    _as_f64_contiguous,
    _compute_lengths,
    _row_chunk_size,
    numba_threads_for,
)


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_gak(X, Y, lens_X, lens_Y, sigma, out):
    n1 = X.shape[0]
    n2 = Y.shape[0]
    d = X.shape[2]
    max_n = Y.shape[1]
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in prange(n1):
        m = lens_X[i]
        prev = numpy.empty(max_n + 1, dtype=numpy.float64)
        curr = numpy.empty(max_n + 1, dtype=numpy.float64)
        for j in range(n2):
            n = lens_Y[j]
            if m == 0 and n == 0:
                out[i, j] = 1.0
                continue
            if m == 0 or n == 0:
                out[i, j] = 0.0
                continue
            prev[0] = 1.0
            for k in range(1, n + 1):
                prev[k] = 0.0
            for ii in range(m):
                curr[0] = 0.0
                for jj in range(n):
                    s = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - Y[j, jj, di]
                        s += diff * diff
                    g = numpy.exp(-s * inv_2sigma2)
                    K = g / (2.0 - g)
                    curr[jj + 1] = (prev[jj + 1] + curr[jj] + prev[jj]) * K
                prev, curr = curr, prev
            out[i, j] = prev[n]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_gak_self(X, lens_X, sigma, row_start, row_stop, out):
    """Self-similarity GAK: diagonal + strict upper triangle, mirrored."""
    n = X.shape[0]
    d = X.shape[2]
    max_n = X.shape[1]
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i_offset in prange(row_stop - row_start):
        i = row_start + i_offset
        m = lens_X[i]
        prev = numpy.empty(max_n + 1, dtype=numpy.float64)
        curr = numpy.empty(max_n + 1, dtype=numpy.float64)
        for j in range(i, n):
            nn = lens_X[j]
            if m == 0 and nn == 0:
                out[i, j] = 1.0
                if i != j:
                    out[j, i] = 1.0
                continue
            if m == 0 or nn == 0:
                out[i, j] = 0.0
                if i != j:
                    out[j, i] = 0.0
                continue
            prev[0] = 1.0
            for k in range(1, nn + 1):
                prev[k] = 0.0
            for ii in range(m):
                curr[0] = 0.0
                for jj in range(nn):
                    s = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - X[j, jj, di]
                        s += diff * diff
                    g = numpy.exp(-s * inv_2sigma2)
                    K = g / (2.0 - g)
                    curr[jj + 1] = (prev[jj + 1] + curr[jj] + prev[jj]) * K
                prev, curr = curr, prev
            v = prev[nn]
            out[i, j] = v
            if i != j:
                out[j, i] = v


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_gak_self_diag(X, lens_X, sigma, out):
    """``out[i] = unnormalized_gak(X[i], X[i], sigma)`` via the same fused
    arithmetic as the cross/self kernels above."""
    n = X.shape[0]
    d = X.shape[2]
    max_n = X.shape[1]
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    for i in prange(n):
        m = lens_X[i]
        if m == 0:
            out[i] = 1.0
            continue
        prev = numpy.empty(max_n + 1, dtype=numpy.float64)
        curr = numpy.empty(max_n + 1, dtype=numpy.float64)
        prev[0] = 1.0
        for k in range(1, m + 1):
            prev[k] = 0.0
        for ii in range(m):
            curr[0] = 0.0
            for jj in range(m):
                s = 0.0
                for di in range(d):
                    diff = X[i, ii, di] - X[i, jj, di]
                    s += diff * diff
                g = numpy.exp(-s * inv_2sigma2)
                K = g / (2.0 - g)
                curr[jj + 1] = (prev[jj + 1] + curr[jj] + prev[jj]) * K
            prev, curr = curr, prev
        out[i] = prev[m]


def cdist_gak_fast(dataset1, dataset2, sigma, n_jobs=None, verbose=0):
    """Fused-kernel entry point. Returns the **un**normalized GAK matrix —
    caller applies the diagonal normalization. ``verbose`` non-zero
    emits ~50 in-place row progress updates."""
    X = _as_f64_contiguous(dataset1)
    with numba_threads_for(n_jobs):
        if dataset2 is None:
            n = X.shape[0]
            out = numpy.empty((n, n), dtype=numpy.float64)
            if n == 0:
                return out
            lens_X = _compute_lengths(X)
            chunk = _row_chunk_size(n, verbose)
            progress = _RowProgress(n, verbose, "cdist_gak")
            try:
                for row_start in range(0, n, chunk):
                    row_stop = min(row_start + chunk, n)
                    _njit_cdist_gak_self(
                        X, lens_X, float(sigma),
                        row_start, row_stop, out,
                    )
                    progress.update(row_stop)
            finally:
                progress.close()
            return out
        Y = _as_f64_contiguous(dataset2)
        out = numpy.empty((X.shape[0], Y.shape[0]), dtype=numpy.float64)
        if X.shape[0] == 0 or Y.shape[0] == 0:
            return out
        if X.shape[2] != Y.shape[2]:
            raise ValueError(
                "XA and XB must have the same number of columns "
                "(i.e. feature dimension.)"
            )
        lens_X = _compute_lengths(X)
        lens_Y = _compute_lengths(Y)
        n = X.shape[0]
        chunk = _row_chunk_size(n, verbose)
        progress = _RowProgress(n, verbose, "cdist_gak")
        try:
            for row_start in range(0, n, chunk):
                row_stop = min(row_start + chunk, n)
                _njit_cdist_gak(
                    X[row_start:row_stop], Y,
                    lens_X[row_start:row_stop], lens_Y, float(sigma),
                    out[row_start:row_stop],
                )
                progress.update(row_stop)
        finally:
            progress.close()
        return out


def gak_self_diag_fast(dataset, sigma):
    X = _as_f64_contiguous(dataset)
    lens_X = _compute_lengths(X)
    out = numpy.empty(X.shape[0], dtype=numpy.float64)
    _njit_gak_self_diag(X, lens_X, float(sigma), out)
    return out

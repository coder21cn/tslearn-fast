"""Fused parallel cdist_frechet kernels for the numpy backend.

Mirror of `_dtw_fast.py`: replaces the per-pair Python wrapper plus per-pair
joblib scheduling in `_cdist_generic` with `@njit(parallel=True)` kernels
that run the discrete Frechet DP inline with `prange` over rows.

Frechet DP recurrence (vs. DTW's ``dist + min``):
    ``acc[i, j] = max(d[i, j]^2, min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1]))``
The result is ``sqrt(acc[l1, l2])`` — same final-cell convention as DTW.
"""
from numba import njit, prange

import numpy

from ._masks import GLOBAL_CONSTRAINT_CODE
from .utils import (
    _RowProgress,
    _as_f64_contiguous,
    _compute_lengths,
    _is_int_valued_finite,
    _resolve_n_jobs,
    _row_chunk_size,
    numba_threads_for,
)

NO_CONSTRAINT = GLOBAL_CONSTRAINT_CODE[None]
SAKOE_CHIBA = GLOBAL_CONSTRAINT_CODE["sakoe_chiba"]


# Selective fastmath: enable reassociation/FMA/etc. but keep ``nnan`` and
# ``ninf`` off so Numba can't assume operands are finite. Plain
# ``fastmath=True`` would let it rewrite ``max(dist, min(...))`` away on
# NaN cells, breaking the parity tests against main's mask path.
_FRECHET_FASTMATH = {"reassoc", "contract", "arcp", "afn", "nsz"}


@njit(fastmath=_FRECHET_FASTMATH, nogil=True, cache=True)
def _frechet_accumulate_cell(dist, acc_up, acc_left, acc_diag):
    """Frechet DP cell update: ``max(dist, min(acc_up, acc_left, acc_diag))``.

    Parameter order matches main's ``_njit_accumulated_matrix`` —
    ``acc_up = acc[i-1, j]``, ``acc_left = acc[i, j-1]``,
    ``acc_diag = acc[i-1, j-1]``. The order is load-bearing under NaN:
    Python/numba ``min(a, b, c)`` returns NaN only when ``a`` is NaN,
    silently skipping later NaN args.
    """
    return max(dist, min(acc_up, acc_left, acc_diag))


@njit(parallel=True, fastmath=_FRECHET_FASTMATH, nogil=True, cache=True)
def _njit_cdist_frechet_unconstrained(X, Y, lens_X, lens_Y, out):
    n1 = X.shape[0]
    n2 = Y.shape[0]
    d = X.shape[2]
    max_sz2 = Y.shape[1]
    INF = numpy.inf

    for i in prange(n1):
        l1 = lens_X[i]
        prev = numpy.empty(max_sz2 + 1, dtype=numpy.float64)
        curr = numpy.empty(max_sz2 + 1, dtype=numpy.float64)
        for j in range(n2):
            l2 = lens_Y[j]
            if l1 == 0 and l2 == 0:
                out[i, j] = 0.0
                continue
            if l1 == 0 or l2 == 0:
                out[i, j] = INF
                continue
            prev[0] = 0.0
            for k in range(1, l2 + 1):
                prev[k] = INF
            for ii in range(l1):
                curr[0] = INF
                for jj in range(l2):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - Y[j, jj, di]
                        dist += diff * diff
                    curr[jj + 1] = _frechet_accumulate_cell(
                        dist, prev[jj + 1], curr[jj], prev[jj]
                    )
                prev, curr = curr, prev
            out[i, j] = numpy.sqrt(prev[l2])


@njit(parallel=True, fastmath=_FRECHET_FASTMATH, nogil=True, cache=True)
def _njit_cdist_frechet_sakoe_chiba(X, Y, lens_X, lens_Y, radius, out):
    n1 = X.shape[0]
    n2 = Y.shape[0]
    d = X.shape[2]
    max_sz2 = Y.shape[1]
    INF = numpy.inf

    for i in prange(n1):
        l1 = lens_X[i]
        prev = numpy.empty(max_sz2 + 1, dtype=numpy.float64)
        curr = numpy.empty(max_sz2 + 1, dtype=numpy.float64)
        for j in range(n2):
            l2 = lens_Y[j]
            if l1 == 0 and l2 == 0:
                out[i, j] = 0.0
                continue
            if l1 == 0 or l2 == 0:
                out[i, j] = INF
                continue
            if l1 <= l2:
                width_lo = radius
                width_hi = l2 - l1 + radius
            else:
                width_lo = l1 - l2 + radius
                width_hi = radius
            prev[0] = 0.0
            for k in range(1, l2 + 1):
                prev[k] = INF
            for k in range(l2 + 1):
                curr[k] = INF
            for ii in range(l1):
                jj_lo = ii - width_lo
                if jj_lo < 0:
                    jj_lo = 0
                jj_hi = ii + width_hi + 1
                if jj_hi > l2:
                    jj_hi = l2
                curr[jj_lo] = INF
                for jj in range(jj_lo, jj_hi):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - Y[j, jj, di]
                        dist += diff * diff
                    curr[jj + 1] = _frechet_accumulate_cell(
                        dist, prev[jj + 1], curr[jj], prev[jj]
                    )
                prev, curr = curr, prev
            out[i, j] = numpy.sqrt(prev[l2])


@njit(parallel=True, fastmath=_FRECHET_FASTMATH, nogil=True, cache=True)
def _njit_cdist_frechet_self_unconstrained(
    X, lens_X, row_start, row_stop, out
):
    n = X.shape[0]
    d = X.shape[2]
    max_sz = X.shape[1]
    INF = numpy.inf

    for i_offset in prange(row_stop - row_start):
        i = row_start + i_offset
        l1 = lens_X[i]
        prev = numpy.empty(max_sz + 1, dtype=numpy.float64)
        curr = numpy.empty(max_sz + 1, dtype=numpy.float64)
        for j in range(i + 1, n):
            l2 = lens_X[j]
            if l1 == 0 and l2 == 0:
                out[i, j] = 0.0
                out[j, i] = 0.0
                continue
            if l1 == 0 or l2 == 0:
                out[i, j] = INF
                out[j, i] = INF
                continue
            prev[0] = 0.0
            for k in range(1, l2 + 1):
                prev[k] = INF
            for ii in range(l1):
                curr[0] = INF
                for jj in range(l2):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - X[j, jj, di]
                        dist += diff * diff
                    curr[jj + 1] = _frechet_accumulate_cell(
                        dist, prev[jj + 1], curr[jj], prev[jj]
                    )
                prev, curr = curr, prev
            v = numpy.sqrt(prev[l2])
            out[i, j] = v
            out[j, i] = v


@njit(parallel=True, fastmath=_FRECHET_FASTMATH, nogil=True, cache=True)
def _njit_cdist_frechet_self_sakoe_chiba(
    X, lens_X, radius, row_start, row_stop, out
):
    n = X.shape[0]
    d = X.shape[2]
    max_sz = X.shape[1]
    INF = numpy.inf

    for i_offset in prange(row_stop - row_start):
        i = row_start + i_offset
        l1 = lens_X[i]
        prev = numpy.empty(max_sz + 1, dtype=numpy.float64)
        curr = numpy.empty(max_sz + 1, dtype=numpy.float64)
        for j in range(i + 1, n):
            l2 = lens_X[j]
            if l1 == 0 and l2 == 0:
                out[i, j] = 0.0
                out[j, i] = 0.0
                continue
            if l1 == 0 or l2 == 0:
                out[i, j] = INF
                out[j, i] = INF
                continue
            if l1 <= l2:
                width_lo = radius
                width_hi = l2 - l1 + radius
            else:
                width_lo = l1 - l2 + radius
                width_hi = radius
            prev[0] = 0.0
            for k in range(1, l2 + 1):
                prev[k] = INF
            for k in range(l2 + 1):
                curr[k] = INF
            for ii in range(l1):
                jj_lo = ii - width_lo
                if jj_lo < 0:
                    jj_lo = 0
                jj_hi = ii + width_hi + 1
                if jj_hi > l2:
                    jj_hi = l2
                curr[jj_lo] = INF
                for jj in range(jj_lo, jj_hi):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - X[j, jj, di]
                        dist += diff * diff
                    curr[jj + 1] = _frechet_accumulate_cell(
                        dist, prev[jj + 1], curr[jj], prev[jj]
                    )
                prev, curr = curr, prev
            v = numpy.sqrt(prev[l2])
            out[i, j] = v
            out[j, i] = v


def cdist_frechet_fast(
    dataset1, dataset2, global_constraint, sakoe_chiba_radius,
    n_jobs=None, verbose=0,
):
    """Fast-path entry. Returns the (n1, n2) cdist Frechet matrix, or
    ``None`` to decline (callers should fall back to ``_cdist_generic``).
    Caller must check that ``global_constraint != "itakura"`` and that
    ``itakura_max_slope is None``. ``dataset2=None`` -> self-similarity:
    strict upper triangle is computed and mirrored, diagonal stays 0.
    ``verbose`` non-zero emits ~50 in-place row progress updates."""
    # Validate ``n_jobs`` upfront — main's joblib-backed path raises
    # on ``n_jobs == 0`` even when no pair would iterate.
    _resolve_n_jobs(n_jobs)
    use_sakoe = global_constraint == SAKOE_CHIBA or (
        global_constraint == NO_CONSTRAINT and sakoe_chiba_radius is not None
    )
    X = _as_f64_contiguous(dataset1)
    lens_X = _compute_lengths(X)
    # No mid-NaN validation: ``acc = max(d^2, min(...))`` swallows NaN
    # cells identically to the legacy ``_njit_frechet`` mask path, so
    # validating here would diverge from main on accepted inputs.

    with numba_threads_for(n_jobs):
        if dataset2 is None:
            n = X.shape[0]
            out = numpy.zeros((n, n), dtype=numpy.float64)
            if n == 0:
                return out
            # Defer to legacy mask path for non-int-valued / non-finite
            # radii: main accepts these and the mask handles them, but our
            # int-bound range kernels can't. Check after empty returns so
            # zero-pair calls do not inspect radius values.
            if (sakoe_chiba_radius is not None
                    and not _is_int_valued_finite(sakoe_chiba_radius)):
                return None
            if use_sakoe:
                # Numba kernels need ``radius`` as int (used in
                # ``range(...)``); main accepts the public float form
                # (e.g. ``1.0``).
                radius = (
                    int(sakoe_chiba_radius)
                    if sakoe_chiba_radius is not None else 1
                )
            chunk = _row_chunk_size(n, verbose)
            progress = _RowProgress(n, verbose, "cdist_frechet")
            try:
                for row_start in range(0, n, chunk):
                    row_stop = min(row_start + chunk, n)
                    if use_sakoe:
                        _njit_cdist_frechet_self_sakoe_chiba(
                            X, lens_X, radius, row_start, row_stop, out
                        )
                    else:
                        _njit_cdist_frechet_self_unconstrained(
                            X, lens_X, row_start, row_stop, out
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
                "All input time series must have the same feature size."
            )
        # Defer to legacy mask path for non-int-valued / non-finite radii:
        # main accepts these and the mask handles them, but our int-bound
        # range kernels can't. Check after empty and feature-dimension
        # short-circuits to preserve public error ordering.
        if (sakoe_chiba_radius is not None
                and not _is_int_valued_finite(sakoe_chiba_radius)):
            return None
        if use_sakoe:
            # Numba kernels need ``radius`` as int (used in ``range(...)``);
            # main accepts the public float form (e.g. ``1.0``).
            radius = (
                int(sakoe_chiba_radius)
                if sakoe_chiba_radius is not None else 1
            )
        lens_Y = _compute_lengths(Y)
        n = X.shape[0]
        chunk = _row_chunk_size(n, verbose)
        progress = _RowProgress(n, verbose, "cdist_frechet")
        try:
            for row_start in range(0, n, chunk):
                row_stop = min(row_start + chunk, n)
                if use_sakoe:
                    _njit_cdist_frechet_sakoe_chiba(
                        X[row_start:row_stop], Y,
                        lens_X[row_start:row_stop], lens_Y, radius,
                        out[row_start:row_stop],
                    )
                else:
                    _njit_cdist_frechet_unconstrained(
                        X[row_start:row_stop], Y,
                        lens_X[row_start:row_stop], lens_Y,
                        out[row_start:row_stop],
                    )
                progress.update(row_stop)
        finally:
            progress.close()
        return out

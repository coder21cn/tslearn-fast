"""Fused parallel cdist_soft_dtw kernels for the numpy backend."""
from numba import njit, prange
import numpy

from .soft_dtw_fast import _njit_soft_dtw, _njit_soft_dtw_grad
from .utils import (
    _RowProgress,
    _as_f64_contiguous,
    _assert_finite_over_valid_prefixes,
    _compute_lengths,
    _row_chunk_size,
)

DBL_MAX = numpy.finfo("double").max


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_soft_dtw(X, Y, lens_X, lens_Y, gamma, out):
    n1 = X.shape[0]
    n2 = Y.shape[0]
    d = X.shape[2]
    max_n = Y.shape[1]

    for i in prange(n1):
        m = lens_X[i]
        if m == 0:
            for j in range(n2):
                out[i, j] = 0.0 if lens_Y[j] == 0 else DBL_MAX
            continue
        D = numpy.empty((m, max_n), dtype=numpy.float64)
        R = numpy.empty((m + 2, max_n + 2), dtype=numpy.float64)
        for j in range(n2):
            n = lens_Y[j]
            if n == 0:
                out[i, j] = DBL_MAX
                continue
            for ii in range(m):
                for jj in range(n):
                    s = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - Y[j, jj, di]
                        s += diff * diff
                    D[ii, jj] = s
            _njit_soft_dtw(D[:m, :n], R[:m + 2, :n + 2], gamma)
            out[i, j] = R[m, n]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_soft_dtw_self(X, lens_X, gamma, row_start, row_stop, out):
    """Self-similarity Soft-DTW: computes diagonal and strict upper triangle,
    mirrors. The diagonal is computed because soft_dtw(x, x) is non-zero.
    """
    n = X.shape[0]
    d = X.shape[2]
    max_m = X.shape[1]

    for i_offset in prange(row_stop - row_start):
        i = row_start + i_offset
        m = lens_X[i]
        if m == 0:
            for j in range(i, n):
                v = 0.0 if lens_X[j] == 0 else DBL_MAX
                out[i, j] = v
                if i != j:
                    out[j, i] = v
            continue
        D = numpy.empty((m, max_m), dtype=numpy.float64)
        R = numpy.empty((m + 2, max_m + 2), dtype=numpy.float64)
        for j in range(i, n):
            nn = lens_X[j]
            if nn == 0:
                out[i, j] = DBL_MAX
                if i != j:
                    out[j, i] = DBL_MAX
                continue
            for ii in range(m):
                for jj in range(nn):
                    s = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - X[j, jj, di]
                        s += diff * diff
                    D[ii, jj] = s
            _njit_soft_dtw(D[:m, :nn], R[:m + 2, :nn + 2], gamma)
            v = R[m, nn]
            out[i, j] = v
            if i != j:
                out[j, i] = v


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_self_soft_dtw_diag(X, lens_X, gamma, out):
    """``out[i] = soft_dtw(X[i], X[i], gamma)``, computed via the same
    arithmetic as the cross/self kernels above so values are bit-equal to
    their diagonal."""
    n = X.shape[0]
    d = X.shape[2]
    for i in prange(n):
        m = lens_X[i]
        if m == 0:
            out[i] = 0.0
            continue
        D = numpy.empty((m, m), dtype=numpy.float64)
        R = numpy.empty((m + 2, m + 2), dtype=numpy.float64)
        for ii in range(m):
            for jj in range(m):
                s = 0.0
                for di in range(d):
                    diff = X[i, ii, di] - X[i, jj, di]
                    s += diff * diff
                D[ii, jj] = s
        _njit_soft_dtw(D[:m, :m], R[:m + 2, :m + 2], gamma)
        out[i] = R[m, m]


def self_soft_dtw_diag_fast(dataset, gamma):
    X = _as_f64_contiguous(dataset)
    lens_X = _compute_lengths(X)
    out = numpy.empty(X.shape[0], dtype=numpy.float64)
    if X.shape[0] == 0:
        return out
    # The fused diagonal kernel doesn't go through SquaredEuclidean, so
    # mid-series NaN/Inf would silently produce NaN entries (and
    # propagate through the normalizer in ``cdist_soft_dtw_normalized``).
    # Match main, which raises via ``SquaredEuclidean.compute``.
    _assert_finite_over_valid_prefixes(X, lens=lens_X)
    _raise_if_any_empty_row(lens_X)
    _njit_self_soft_dtw_diag(X, lens_X, float(gamma), out)
    return out


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_softdtw_obj_grad(Z, X, lens_X, gamma, value_per_i, G_per_i):
    """Per-i Soft-DTW barycenter objective + gradient. Computes
    ``value_per_i[i] = soft_dtw(Z, X[i])`` and
    ``G_per_i[i] = 2 * (Z * E.sum(1) - E @ X[i])`` where ``E`` is the
    soft-DTW gradient matrix. Caller weights and reduces across ``i``.

    Z : (m, d) float64
    X : (n_X, max_n, d) float64
    lens_X : (n_X,) int64
    gamma : float64
    value_per_i : (n_X,) float64 — output
    G_per_i : (n_X, m, d) float64 — output
    """
    n_X = X.shape[0]
    m = Z.shape[0]
    d = Z.shape[1]

    for i in prange(n_X):
        n_i = lens_X[i]
        if n_i == 0:
            value_per_i[i] = 0.0
            for r in range(m):
                for c in range(d):
                    G_per_i[i, r, c] = 0.0
            continue

        # D extended with a zero last row/col (`_njit_soft_dtw_grad` requires
        # shape (m+1, n_i+1)).
        D = numpy.zeros((m + 1, n_i + 1), dtype=numpy.float64)
        for ii in range(m):
            for jj in range(n_i):
                s = 0.0
                for di in range(d):
                    diff = Z[ii, di] - X[i, jj, di]
                    s += diff * diff
                D[ii, jj] = s

        R = numpy.empty((m + 2, n_i + 2), dtype=numpy.float64)
        _njit_soft_dtw(D[:m, :n_i], R, gamma)
        value_per_i[i] = R[m, n_i]

        E = numpy.zeros((m + 2, n_i + 2), dtype=numpy.float64)
        _njit_soft_dtw_grad(D, R, E, gamma)

        # Jacobian product for squared euclidean: G[ii, k] =
        #   2 * (Z[ii, k] * sum_j E[ii+1, j+1] - sum_j E[ii+1, j+1] * X[i, j, k])
        for ii in range(m):
            row_sum = 0.0
            for jj in range(n_i):
                row_sum += E[ii + 1, jj + 1]
            for k in range(d):
                acc = 0.0
                for jj in range(n_i):
                    acc += E[ii + 1, jj + 1] * X[i, jj, k]
                G_per_i[i, ii, k] = 2.0 * (Z[ii, k] * row_sum - acc)


def softdtw_obj_grad_fast(Z, X_padded, lens_X, weights, gamma):
    """Compute ``(obj, G)`` for the Soft-DTW barycenter objective. ``X_padded``
    is the (n_X, max_n, d) tensor of pre-trimmed series stacked with NaN/zero
    padding; ``lens_X`` are the valid lengths. Reduction is done in numpy
    after the parallel kernel."""
    Z = numpy.ascontiguousarray(Z, dtype=numpy.float64)
    n_X = X_padded.shape[0]
    m = Z.shape[0]
    d = Z.shape[1]
    value_per_i = numpy.empty(n_X, dtype=numpy.float64)
    G_per_i = numpy.empty((n_X, m, d), dtype=numpy.float64)
    _njit_softdtw_obj_grad(
        Z, X_padded, lens_X, float(gamma), value_per_i, G_per_i
    )
    obj = float(numpy.dot(weights, value_per_i))
    G = (weights[:, None, None] * G_per_i).sum(axis=0)
    return obj, G


def cdist_soft_dtw_fast(dataset1, dataset2, gamma, verbose=0):
    """Returns the cdist_soft_dtw matrix.

    Parameters
    ----------
    dataset1 : numpy.ndarray, shape (n1, max_sz1, d)
    dataset2 : numpy.ndarray or None, shape (n2, max_sz2, d)
    gamma : float
    verbose : int, optional (default=0)
        Non-zero emits ~50 in-place row progress updates on stderr.
    """
    X = _as_f64_contiguous(dataset1)
    # Per-pair validation that ``SquaredEuclidean.compute`` does on
    # main is replaced by an upfront prefix scan, but only on the
    # iterating side — the empty-cross case must short-circuit to
    # match main, which never inspects either side when no pair runs.
    if dataset2 is None:
        n = X.shape[0]
        out = numpy.empty((n, n), dtype=numpy.float64)
        if n == 0:
            return out
        lens_X = _compute_lengths(X)
        _assert_finite_over_valid_prefixes(X, lens=lens_X)
        _raise_if_any_empty_row(lens_X)
        chunk = _row_chunk_size(n, verbose)
        progress = _RowProgress(n, verbose, "cdist_soft_dtw")
        try:
            for row_start in range(0, n, chunk):
                row_stop = min(row_start + chunk, n)
                _njit_cdist_soft_dtw_self(
                    X, lens_X, float(gamma),
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
            "Incompatible dimension for X and Y matrices: "
            f"X.shape[2] == {X.shape[2]} while Y.shape[2] == {Y.shape[2]}"
        )
    lens_X = _compute_lengths(X)
    lens_Y = _compute_lengths(Y)
    _assert_finite_over_valid_prefixes(X, lens=lens_X)
    _assert_finite_over_valid_prefixes(Y, lens=lens_Y)
    _raise_if_any_empty_row(lens_X)
    _raise_if_any_empty_row(lens_Y)
    n = X.shape[0]
    chunk = _row_chunk_size(n, verbose)
    progress = _RowProgress(n, verbose, "cdist_soft_dtw")
    try:
        for row_start in range(0, n, chunk):
            row_stop = min(row_start + chunk, n)
            _njit_cdist_soft_dtw(
                X[row_start:row_stop], Y,
                lens_X[row_start:row_stop], lens_Y, float(gamma),
                out[row_start:row_stop],
            )
            progress.update(row_stop)
    finally:
        progress.close()
    return out


def _raise_if_any_empty_row(lens):
    """Reject zero-length / all-NaN-padded series on the Soft-DTW fast
    path. Main trims these to shape ``(0, d)`` and
    ``SquaredEuclidean.compute`` raises via sklearn's ``check_array``;
    the fused kernel quietly emits 0 / DBL_MAX without the helper."""
    if int(lens.min()) == 0:
        raise ValueError(
            "Soft-DTW input contains a zero-length / all-NaN series; "
            "main raises via SquaredEuclidean.compute."
        )

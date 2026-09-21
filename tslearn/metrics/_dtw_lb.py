"""LB_Keogh + early-abandon DTW for fast kNN-style top-k DTW queries.

This is a UCR-Suite-style fast path: precompute the upper/lower envelope of
each candidate series once, then for every query, prune candidates whose
LB_Keogh against the envelope already exceeds the running k-th best DTW
distance, and run early-abandon DTW on the rest.

Multivariate generalization: per-dimension LB_Keogh values are summed to
form a (looser but valid) lower bound on multivariate DTW. The cell cost
is the across-dim sum of squared differences, matching the standard
multivariate DTW definition.

Constraints:

- Numpy backend only.
- Equal valid length across query and candidate (typical UCR setting).
- Sakoe-Chiba band (the tightest, most common DTW constraint).

The dispatcher returns ``None`` when these don't hold; callers fall back
to the regular ``cdist_dtw_fast`` path.
"""
from numba import njit, prange

import numpy

from .utils import _as_f64_contiguous, _compute_lengths, numba_threads_for


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_envelopes(X, lens, radius, U, L):
    """Per-series, per-dim moving max/min envelope. ``X`` is
    ``(n, max_sz, d)``; ``U`` and ``L`` are ``(n, max_sz, d)`` buffers
    written in place over the valid prefix ``X[i, :lens[i]]``."""
    n = X.shape[0]
    d = X.shape[2]
    INF = numpy.inf
    for i in prange(n):
        sz = lens[i]
        for t in range(sz):
            lo = t - radius
            hi = t + radius + 1
            if lo < 0:
                lo = 0
            if hi > sz:
                hi = sz
            for di in range(d):
                mx = -INF
                mn = INF
                for s in range(lo, hi):
                    v = X[i, s, di]
                    if v > mx:
                        mx = v
                    if v < mn:
                        mn = v
                U[i, t, di] = mx
                L[i, t, di] = mn


@njit(fastmath=True, nogil=True, inline="always", cache=True)
def _lb_keogh_sq(q, U, L, sz, d):
    """Sum across dims of per-dim LB_Keogh-squared between ``q[:sz]`` and
    envelope ``(U[:sz], L[:sz])``. Valid lower bound on multivariate DTW."""
    s = 0.0
    for t in range(sz):
        for di in range(d):
            v = q[t, di]
            u = U[t, di]
            if v > u:
                diff = v - u
                s += diff * diff
            else:
                lo = L[t, di]
                if v < lo:
                    diff = v - lo
                    s += diff * diff
    return s


@njit(fastmath=True, nogil=True, cache=True)
def _dtw_sakoe_ea(s1, s2, l1, l2, d, width_lo, width_hi, ub_sq, prev, curr):
    """Sakoe-Chiba DTW with early abandon. Returns the SQUARED DTW distance,
    or ``+inf`` if every cell of some row exceeded ``ub_sq``. Multivariate:
    cell cost is the across-dim sum of squared differences.

    ``prev`` and ``curr`` are pre-allocated rolling buffers of length
    ``max_l2 + 1`` (caller provides; reused across calls)."""
    INF = numpy.inf
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
        row_min = INF
        for jj in range(jj_lo, jj_hi):
            dist = 0.0
            for di in range(d):
                diff = s1[ii, di] - s2[jj, di]
                dist += diff * diff
            a = prev[jj]
            b = prev[jj + 1]
            c = curr[jj]
            m = a if a < b else b
            if c < m:
                m = c
            v = dist + m
            curr[jj + 1] = v
            if v < row_min:
                row_min = v
        if row_min >= ub_sq:
            return INF
        # rolling-buffer pointer swap
        tmp = prev
        prev = curr
        curr = tmp
    return prev[l2]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_topk_sakoe(
    X, Y, lens_X, lens_Y, U_Y, L_Y, radius, k, dists_out, idx_out
):
    """For each ``X[i]`` find the ``k`` closest ``Y[j]`` under DTW with a
    Sakoe-Chiba band. LB_Keogh prunes against the envelope of each Y
    candidate; per-dim LB sums for d>1 (looser but valid bound).

    ``dists_out``/``idx_out`` are ``(n_X, k)`` buffers. Output is sorted
    ascending per row; unfilled slots are ``+inf`` / ``-1``.
    """
    n1 = X.shape[0]
    n2 = Y.shape[0]
    max_sz = Y.shape[1]
    d = X.shape[2]
    INF = numpy.inf

    for i in prange(n1):
        l1 = lens_X[i]
        prev = numpy.empty(max_sz + 1, dtype=numpy.float64)
        curr = numpy.empty(max_sz + 1, dtype=numpy.float64)
        # Row's running top-k stored as parallel sorted arrays:
        # top_d[0..k] ascending squared distances, top_i[0..k] candidate idx.
        top_d = numpy.empty(k, dtype=numpy.float64)
        top_i = numpy.empty(k, dtype=numpy.int64)
        for t in range(k):
            top_d[t] = INF
            top_i[t] = -1
        if l1 == 0:
            for t in range(k):
                dists_out[i, t] = INF
                idx_out[i, t] = -1
            continue
        for j in range(n2):
            l2 = lens_Y[j]
            if l2 == 0 or l1 != l2:
                continue
            ub_sq = top_d[k - 1]  # current k-th best (squared distance)
            lb_sq = _lb_keogh_sq(X[i], U_Y[j], L_Y[j], l1, d)
            if lb_sq >= ub_sq:
                continue
            d_sq = _dtw_sakoe_ea(
                X[i], Y[j], l1, l2, d, radius, radius, ub_sq, prev, curr
            )
            if d_sq < ub_sq:
                pos = k - 1
                while pos > 0 and top_d[pos - 1] > d_sq:
                    top_d[pos] = top_d[pos - 1]
                    top_i[pos] = top_i[pos - 1]
                    pos -= 1
                top_d[pos] = d_sq
                top_i[pos] = j
        for t in range(k):
            v = top_d[t]
            if v == INF:
                dists_out[i, t] = INF
            else:
                dists_out[i, t] = numpy.sqrt(v)
            idx_out[i, t] = top_i[t]


def cdist_dtw_topk_fast(
    dataset_query, dataset_candidates, k, radius, n_jobs=None
):
    """Top-k DTW search via LB_Keogh + early-abandon, or ``None`` if
    fast-path constraints are not met.

    Parameters
    ----------
    dataset_query : numpy.ndarray, shape (n_query, max_sz, d)
    dataset_candidates : numpy.ndarray, shape (n_cand, max_sz, d)
    k : int
        Number of neighbors per query.
    radius : int
        Sakoe-Chiba radius. Must be > 0; ``None`` returns ``None`` (caller
        falls back to the no-LB fast path).
    n_jobs : int or None
        Same semantics as ``cdist_dtw_fast``.

    Returns
    -------
    (dists, indices) : tuple of numpy.ndarray, both shape (n_query, k)
        Sorted ascending; padded with ``+inf`` / ``-1`` when fewer than ``k``
        candidates pass.
    None
        If the inputs don't satisfy the fast-path constraints (equal valid
        length across all series, ``radius`` set, matching dim), or distance
        overflow prevents filling the available neighbor slots.
    """
    if radius is None or radius < 0:
        return None
    X = _as_f64_contiguous(dataset_query)
    Y = _as_f64_contiguous(dataset_candidates)
    if X.ndim != 3 or Y.ndim != 3:
        return None
    if X.shape[2] != Y.shape[2] or X.shape[2] == 0:
        return None
    if X.shape[0] == 0 or Y.shape[0] == 0:
        n_query = X.shape[0]
        return (
            numpy.full((n_query, k), numpy.inf, dtype=numpy.float64),
            numpy.full((n_query, k), -1, dtype=numpy.int64),
        )
    lens_X = _compute_lengths(X)
    lens_Y = _compute_lengths(Y)
    # LB_Keogh requires equal-length series (the envelope is per-position),
    # so we need uniform lengths across X and Y. Variable-length padded
    # datasets fall back to the no-LB fast path; the alternative — letting
    # the kernel skip ``l1 != l2`` candidates per-pair — would silently
    # return INF for queries whose length matches no candidate, which
    # diverges from the slow path's any-length DTW semantics.
    if not (numpy.all(lens_X == lens_X[0]) and numpy.all(lens_Y == lens_Y[0])
            and lens_X[0] == lens_Y[0] and lens_X[0] > 0):
        return None
    # LB_Keogh + early-abandon prunes NaN candidates (``NaN < ub_sq``
    # is False), so a non-finite valid prefix would silently lose the
    # real nearest neighbor — fall back to the full cdist path.
    L_valid = int(lens_X[0])
    if not (numpy.isfinite(X[:, :L_valid]).all()
            and numpy.isfinite(Y[:, :L_valid]).all()):
        return None

    with numba_threads_for(n_jobs):
        max_sz_y = Y.shape[1]
        d = Y.shape[2]
        U = numpy.empty((Y.shape[0], max_sz_y, d), dtype=numpy.float64)
        L = numpy.empty((Y.shape[0], max_sz_y, d), dtype=numpy.float64)
        _njit_envelopes(Y, lens_Y, int(radius), U, L)
        dists = numpy.empty((X.shape[0], k), dtype=numpy.float64)
        idxs = numpy.empty((X.shape[0], k), dtype=numpy.int64)
        _njit_cdist_dtw_topk_sakoe(
            X, Y, lens_X, lens_Y, U, L, int(radius), int(k), dists, idxs
        )
        # Finite inputs can overflow squared distances. Pruning then leaves
        # -1 slots even with enough candidates; let the full-distance path
        # rank those infinite distances instead. Ignore intentional padding
        # beyond the candidate count when k is oversized.
        if (idxs[:, :min(k, Y.shape[0])] < 0).any():
            return None
        return dists, idxs

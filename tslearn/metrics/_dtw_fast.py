"""Fused parallel cdist_dtw kernels for the numpy backend."""
from numba import njit, prange

import numpy

from ._masks import GLOBAL_CONSTRAINT_CODE, _njit_itakura_mask
from .utils import (
    _RowProgress,
    _as_f64_contiguous,
    _compute_lengths,
    _is_int_valued_finite,
    _njit_compute_path,
    _raise_if_ambiguous_constraint,
    _resolve_n_jobs,
    _row_chunk_size,
    numba_threads_for,
)

NO_CONSTRAINT = GLOBAL_CONSTRAINT_CODE[None]
ITAKURA = GLOBAL_CONSTRAINT_CODE["itakura"]
SAKOE_CHIBA = GLOBAL_CONSTRAINT_CODE["sakoe_chiba"]


def _itakura_row_bounds(sz1, sz2, max_slope):
    """Per-row ``[j_lo, j_hi)`` bounds derived from the Itakura parallelogram
    mask. The mask is contiguous along each row (convex parallelogram), so
    each row is fully described by a single open interval. Rows that fall
    outside the parallelogram (no admissible ``j``) yield ``j_lo == j_hi``;
    ``_njit_itakura_mask`` raises before we get here when the configuration
    is fully infeasible."""
    mask = _njit_itakura_mask(sz1, sz2, max_slope=max_slope)
    # numpy.argmax on bool returns the first True; an all-False row gives 0.
    j_lo = mask.argmax(axis=1).astype(numpy.int64)
    last = sz2 - mask[:, ::-1].argmax(axis=1).astype(numpy.int64)
    has_any = mask.any(axis=1)
    j_lo = numpy.where(has_any, j_lo, 0)
    j_hi = numpy.where(has_any, last, 0)
    return j_lo, j_hi


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_unconstrained(X, Y, lens_X, lens_Y, out):
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
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            out[i, j] = numpy.sqrt(prev[l2])


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_sakoe_chiba(X, Y, lens_X, lens_Y, radius, out):
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
            # Match `_sakoe_chiba_mask`: band extends by |l1 - l2| on the longer axis.
            if l1 <= l2:
                width_lo = radius
                width_hi = l2 - l1 + radius
            else:
                width_lo = l1 - l2 + radius
                width_hi = radius
            # Full INF init of both rolling rows once per (i, j); per-row we
            # only have to refresh the left band boundary in `curr`. Cells
            # outside any row's band stay INF from this initial fill.
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
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            out[i, j] = numpy.sqrt(prev[l2])


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_self_unconstrained(X, lens_X, row_start, row_stop, out):
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
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            v = numpy.sqrt(prev[l2])
            out[i, j] = v
            out[j, i] = v


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_self_sakoe_chiba(
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
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            v = numpy.sqrt(prev[l2])
            out[i, j] = v
            out[j, i] = v


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_itakura(X, Y, l1, l2, j_lo, j_hi, out):
    """Itakura-parallelogram cdist for equal-valid-length numpy data. ``l1``
    is the common valid length within ``X`` and ``l2`` the common valid
    length within ``Y``; ``j_lo[i]``/``j_hi[i]`` are the per-row bounds for
    one (l1, l2) parallelogram, shared across all (i, j) pairs."""
    n1 = X.shape[0]
    n2 = Y.shape[0]
    d = X.shape[2]
    INF = numpy.inf

    for i in prange(n1):
        prev = numpy.empty(l2 + 1, dtype=numpy.float64)
        curr = numpy.empty(l2 + 1, dtype=numpy.float64)
        for j in range(n2):
            prev[0] = 0.0
            for k in range(1, l2 + 1):
                prev[k] = INF
            for k in range(l2 + 1):
                curr[k] = INF
            for ii in range(l1):
                jj_lo = j_lo[ii]
                jj_hi = j_hi[ii]
                if jj_lo >= jj_hi:
                    prev, curr = curr, prev
                    continue
                curr[jj_lo] = INF
                for jj in range(jj_lo, jj_hi):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - Y[j, jj, di]
                        dist += diff * diff
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            out[i, j] = numpy.sqrt(prev[l2])


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _njit_cdist_dtw_self_itakura(
    X, l_eq, j_lo, j_hi, row_start, row_stop, out
):
    """Self-similarity Itakura cdist (square parallelogram, ``l1 == l2 == l_eq``)."""
    n = X.shape[0]
    d = X.shape[2]
    INF = numpy.inf

    for i_offset in prange(row_stop - row_start):
        i = row_start + i_offset
        prev = numpy.empty(l_eq + 1, dtype=numpy.float64)
        curr = numpy.empty(l_eq + 1, dtype=numpy.float64)
        for j in range(i + 1, n):
            prev[0] = 0.0
            for k in range(1, l_eq + 1):
                prev[k] = INF
            for k in range(l_eq + 1):
                curr[k] = INF
            for ii in range(l_eq):
                jj_lo = j_lo[ii]
                jj_hi = j_hi[ii]
                if jj_lo >= jj_hi:
                    prev, curr = curr, prev
                    continue
                curr[jj_lo] = INF
                for jj in range(jj_lo, jj_hi):
                    dist = 0.0
                    for di in range(d):
                        diff = X[i, ii, di] - X[j, jj, di]
                        dist += diff * diff
                    a = prev[jj]
                    b = prev[jj + 1]
                    c = curr[jj]
                    m = a if a < b else b
                    if c < m:
                        m = c
                    curr[jj + 1] = dist + m
                prev, curr = curr, prev
            v = numpy.sqrt(prev[l_eq])
            out[i, j] = v
            out[j, i] = v


@njit(fastmath=True, nogil=True, cache=True)
def _njit_dtw_sakoe(s1, s2, radius):
    """Single-pair DTW distance with Sakoe-Chiba band iteration. Returns
    ``sqrt(cost)``. Path traceback is unnecessary, so we keep only two
    rolling rows."""
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    d = s1.shape[1]
    INF = numpy.inf

    if l1 <= l2:
        width_lo = radius
        width_hi = l2 - l1 + radius
    else:
        width_lo = l1 - l2 + radius
        width_hi = radius

    prev = numpy.empty(l2 + 1, dtype=numpy.float64)
    curr = numpy.empty(l2 + 1, dtype=numpy.float64)
    prev[0] = 0.0
    for k in range(1, l2 + 1):
        prev[k] = INF
    for k in range(l2 + 1):
        curr[k] = INF
    for i in range(l1):
        jj_lo = i - width_lo
        if jj_lo < 0:
            jj_lo = 0
        jj_hi = i + width_hi + 1
        if jj_hi > l2:
            jj_hi = l2
        curr[jj_lo] = INF
        for j in range(jj_lo, jj_hi):
            dist = 0.0
            for di in range(d):
                diff = s1[i, di] - s2[j, di]
                dist += diff * diff
            a = prev[j]
            b = prev[j + 1]
            c = curr[j]
            m = a if a < b else b
            if c < m:
                m = c
            curr[j + 1] = dist + m
        prev, curr = curr, prev
    return numpy.sqrt(prev[l2])


@njit(fastmath=True, nogil=True, cache=True)
def _njit_dtw_path_sakoe(s1, s2, radius):
    """Single-pair DTW path with Sakoe-Chiba band iteration.

    Returns ``(sqrt(cost), path)`` matching the legacy internal
    ``_njit_dtw_path`` contract (the public wrapper swaps the order).
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    d = s1.shape[1]
    INF = numpy.inf

    if l1 <= l2:
        width_lo = radius
        width_hi = l2 - l1 + radius
    else:
        width_lo = l1 - l2 + radius
        width_hi = radius

    cum_sum = numpy.full((l1 + 1, l2 + 1), INF)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        jj_lo = i - width_lo
        if jj_lo < 0:
            jj_lo = 0
        jj_hi = i + width_hi + 1
        if jj_hi > l2:
            jj_hi = l2
        for j in range(jj_lo, jj_hi):
            dist = 0.0
            for di in range(d):
                diff = s1[i, di] - s2[j, di]
                dist += diff * diff
            a = cum_sum[i, j]
            b = cum_sum[i, j + 1]
            c = cum_sum[i + 1, j]
            m = a if a < b else b
            if c < m:
                m = c
            cum_sum[i + 1, j + 1] = dist + m

    path = _njit_compute_path(cum_sum[1:, 1:])
    return numpy.sqrt(cum_sum[l1, l2]), path


@njit(fastmath=True, nogil=True, cache=True)
def _njit_dtw_path_sakoe_arr(s1, s2, radius):
    """Same as ``_njit_dtw_path_sakoe`` but returns the path as a
    ``(path_len, 2)`` int64 ndarray instead of a numba reflected list
    of tuples.

    Used by the DBA assignment loop, where we'd otherwise pay a per-
    call ``numpy.asarray(path, dtype=int64)`` conversion that turned
    out to be the dominant DBA cost on small problems.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    d = s1.shape[1]
    INF = numpy.inf

    if l1 <= l2:
        width_lo = radius
        width_hi = l2 - l1 + radius
    else:
        width_lo = l1 - l2 + radius
        width_hi = radius

    cum_sum = numpy.full((l1 + 1, l2 + 1), INF)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        jj_lo = i - width_lo
        if jj_lo < 0:
            jj_lo = 0
        jj_hi = i + width_hi + 1
        if jj_hi > l2:
            jj_hi = l2
        for j in range(jj_lo, jj_hi):
            dist = 0.0
            for di in range(d):
                diff = s1[i, di] - s2[j, di]
                dist += diff * diff
            a = cum_sum[i, j]
            b = cum_sum[i, j + 1]
            c = cum_sum[i + 1, j]
            m = a if a < b else b
            if c < m:
                m = c
            cum_sum[i + 1, j + 1] = dist + m

    # Backtrack into a pre-allocated ``(l1+l2, 2)`` buffer (path length
    # is at most ``l1 + l2 - 1``); fill from the back, return the
    # populated tail slice.
    path = numpy.empty((l1 + l2, 2), dtype=numpy.int64)
    pos = path.shape[0] - 1
    i = l1 - 1
    j = l2 - 1
    path[pos, 0] = i
    path[pos, 1] = j
    pos -= 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a = cum_sum[i, j]
            b = cum_sum[i, j + 1]
            c = cum_sum[i + 1, j]
            if a <= b and a <= c:
                i -= 1
                j -= 1
            elif b <= c:
                i -= 1
            else:
                j -= 1
        path[pos, 0] = i
        path[pos, 1] = j
        pos -= 1
    return numpy.sqrt(cum_sum[l1, l2]), path[pos + 1:]


def cdist_dtw_fast(
    dataset1, dataset2, global_constraint, sakoe_chiba_radius,
    itakura_max_slope=None, n_jobs=None, verbose=0
):
    """Returns the cdist matrix, or ``None`` if the inputs aren't supported
    by the fast path (currently: Itakura on variable-length data within a
    dataset). Callers should fall back to ``_cdist_generic`` when ``None``
    is returned.

    Parameters
    ----------
    dataset1 : numpy.ndarray, shape (n1, max_sz1, d)
        Padded with NaN for variable-length series.
    dataset2 : numpy.ndarray or None, shape (n2, max_sz2, d)
        If None, self-similarity of dataset1.
    global_constraint : int
        ``GLOBAL_CONSTRAINT_CODE[None | "sakoe_chiba" | "itakura"]``.
    sakoe_chiba_radius : int or None
        Defaults to 1 when ``global_constraint`` is ``"sakoe_chiba"`` and this
        is None.
    itakura_max_slope : float or None
        Defaults to 2.0 when ``global_constraint`` is ``"itakura"`` and this
        is None.
    n_jobs : int or None
        Mapped to numba's thread count for the duration of the call. ``None``
        and 1 -> 1 thread (sklearn convention); positive int -> that many; -1
        -> numba default. ``joblib.parallel_backend`` contexts are not observed.
    verbose : int, optional (default=0)
        If non-zero, report roughly 50 in-place row progress updates for the
        numba fast path.
    """
    # Validate ``n_jobs`` upfront — main's joblib-backed path raises
    # on ``n_jobs == 0`` even when no pair would iterate.
    _resolve_n_jobs(n_jobs)
    X = _as_f64_contiguous(dataset1)
    lens_X = _compute_lengths(X)
    if dataset2 is None:
        Y = None
        lens_Y = None
    else:
        Y = _as_f64_contiguous(dataset2)
        lens_Y = _compute_lengths(Y)
    # Short-circuit empty shapes before raising on ambiguous params:
    # main raises only per-pair via ``_compute_mask`` and returns an
    # empty matrix when no pair iterates.
    if Y is None and X.shape[0] == 0:
        return numpy.zeros((0, 0), dtype=numpy.float64)
    if Y is not None and (X.shape[0] == 0 or Y.shape[0] == 0):
        return numpy.empty((X.shape[0], Y.shape[0]), dtype=numpy.float64)
    if Y is not None and X.shape[2] != Y.shape[2]:
        raise ValueError(
            "All input time series must have the same feature size."
        )
    _raise_if_ambiguous_constraint(
        global_constraint, sakoe_chiba_radius, itakura_max_slope,
    )
    # Defer to legacy mask path for negative / non-int-valued / non-finite radii:
    # main accepts these and the mask handles them, but our int-bound
    # range kernels can't.
    if (sakoe_chiba_radius is not None
            and (not _is_int_valued_finite(sakoe_chiba_radius)
                 or sakoe_chiba_radius < 0)):
        return None
    use_sakoe = global_constraint == SAKOE_CHIBA or (
        global_constraint == NO_CONSTRAINT and sakoe_chiba_radius is not None
    )
    use_itakura = global_constraint == ITAKURA or (
        global_constraint == NO_CONSTRAINT and itakura_max_slope is not None
        and not use_sakoe
    )
    if use_sakoe:
        # Numba kernels use ``radius`` in ``range(...)`` bounds, so
        # the public-facing float values main accepts (e.g. ``1.0``)
        # need int normalization.
        radius = int(sakoe_chiba_radius) if sakoe_chiba_radius is not None else 1
    if use_itakura:
        slope = itakura_max_slope if itakura_max_slope is not None else 2.0
        # Fast path needs a single shared parallelogram per call, which
        # requires equal valid length within each dataset.
        if not numpy.all(lens_X == lens_X[0]):
            return None
        if lens_Y is not None and not numpy.all(lens_Y == lens_Y[0]):
            return None
        # Preserve legacy Itakura behavior on zero-valid-length series:
        # the mask construction raises (division by zero / infeasible mask)
        # instead of returning numeric 0/inf distances.
        if lens_X[0] == 0 or (lens_Y is not None and lens_Y[0] == 0):
            return None

    with numba_threads_for(n_jobs):
        if Y is None:
            n = X.shape[0]
            out = numpy.zeros((n, n), dtype=numpy.float64)
            chunk = _row_chunk_size(n, verbose)
            progress = _RowProgress(n, verbose, "cdist_dtw")
            try:
                if use_sakoe:
                    for row_start in range(0, n, chunk):
                        row_stop = min(row_start + chunk, n)
                        _njit_cdist_dtw_self_sakoe_chiba(
                            X, lens_X, radius, row_start, row_stop, out
                        )
                        progress.update(row_stop)
                elif use_itakura:
                    l_eq = int(lens_X[0])
                    j_lo, j_hi = _itakura_row_bounds(l_eq, l_eq, slope)
                    for row_start in range(0, n, chunk):
                        row_stop = min(row_start + chunk, n)
                        _njit_cdist_dtw_self_itakura(
                            X, l_eq, j_lo, j_hi, row_start, row_stop, out
                        )
                        progress.update(row_stop)
                else:
                    for row_start in range(0, n, chunk):
                        row_stop = min(row_start + chunk, n)
                        _njit_cdist_dtw_self_unconstrained(
                            X, lens_X, row_start, row_stop, out
                        )
                        progress.update(row_stop)
            finally:
                progress.close()
            return out

        out = numpy.empty((X.shape[0], Y.shape[0]), dtype=numpy.float64)
        n = X.shape[0]
        chunk = _row_chunk_size(n, verbose)
        progress = _RowProgress(n, verbose, "cdist_dtw")
        try:
            if use_sakoe:
                for row_start in range(0, n, chunk):
                    row_stop = min(row_start + chunk, n)
                    _njit_cdist_dtw_sakoe_chiba(
                        X[row_start:row_stop],
                        Y,
                        lens_X[row_start:row_stop],
                        lens_Y,
                        radius,
                        out[row_start:row_stop],
                    )
                    progress.update(row_stop)
            elif use_itakura:
                l_eq_X = int(lens_X[0])
                l_eq_Y = int(lens_Y[0])
                j_lo, j_hi = _itakura_row_bounds(l_eq_X, l_eq_Y, slope)
                for row_start in range(0, n, chunk):
                    row_stop = min(row_start + chunk, n)
                    _njit_cdist_dtw_itakura(
                        X[row_start:row_stop],
                        Y,
                        l_eq_X,
                        l_eq_Y,
                        j_lo,
                        j_hi,
                        out[row_start:row_stop],
                    )
                    progress.update(row_stop)
            else:
                for row_start in range(0, n, chunk):
                    row_stop = min(row_start + chunk, n)
                    _njit_cdist_dtw_unconstrained(
                        X[row_start:row_stop],
                        Y,
                        lens_X[row_start:row_stop],
                        lens_Y,
                        out[row_start:row_stop],
                    )
                    progress.update(row_stop)
        finally:
            progress.close()
        return out

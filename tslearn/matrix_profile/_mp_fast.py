"""STOMP-style matrix-profile kernel for the numpy backend.

For a single (multivariate) time series, computes the minimum squared
Euclidean distance from each subsequence to its nearest non-trivial-match
subsequence (``|i - j| > band_width``). Uses STOMP's diagonal dot-product
recurrence — O(n^2 * d) total work vs. the naive O(n^2 * m * d).

Supports both unscaled (plain Euclidean) and z-normalized distance, and
both univariate and multivariate inputs.
"""
from numba import njit, prange, get_num_threads, get_thread_id
import numpy


@njit(nogil=True, cache=True)
def _scaled_distance_sq(X, i, j, m):
    """Stable z-normalized distance for a cancellation-prone window pair."""
    dist_sq = 0.0
    for di in range(X.shape[1]):
        mu_i = 0.0
        mu_j = 0.0
        for k in range(m):
            mu_i += X[i + k, di]
            mu_j += X[j + k, di]
        mu_i /= m
        mu_j /= m
        var_i = 0.0
        var_j = 0.0
        for k in range(m):
            var_i += (X[i + k, di] - mu_i) ** 2
            var_j += (X[j + k, di] - mu_j) ** 2
        sig_i = numpy.sqrt(var_i / m)
        sig_j = numpy.sqrt(var_j / m)
        if sig_i == 0.0:
            sig_i = 1.0
        if sig_j == 0.0:
            sig_j = 1.0
        for k in range(m):
            diff = ((X[i + k, di] - mu_i) / sig_i
                    - (X[j + k, di] - mu_j) / sig_j)
            dist_sq += diff * diff
    return dist_sq


def _run_mp_stomp(X, m, scale, band_width, mins_sq):
    """STOMP matrix profile for a single multivariate time series.

    Wrapper that allocates the per-thread reduction buffer. ``get_num_threads``
    is called here (outside the jitted kernel) so the buffer is sized before
    the parallel region; ``local_mins`` and ``n_threads`` are passed in as
    plain args. ``get_thread_id`` must remain inside the kernel body (it
    resolves per-iteration inside ``prange``) so ``cache=True`` is not used
    on the inner kernel.

    Parameters
    ----------
    X : (sz, d) float64, contiguous
    m : int — subsequence length.
    scale : bool — if True, z-normalize each subsequence (per-dim) before
        distance. If False, plain Euclidean on the raw signal.
    band_width : int — exclusion zone half-width; pairs with
        ``|i - j| <= band_width`` are skipped.
    mins_sq : (n,) float64 — output, written in place. ``n = sz - m + 1``.
    """
    n = X.shape[0] - m + 1
    if n <= 1:
        # Match the no-op fast-return inside the kernel; avoids allocating
        # the per-thread buffer at all when there's nothing to reduce.
        for i in range(max(n, 0)):
            mins_sq[i] = numpy.inf
        return
    n_threads = get_num_threads()
    local_mins = numpy.full((n_threads, n), numpy.inf, dtype=numpy.float64)
    _njit_mp_stomp_kernel(
        X, m, scale, band_width, mins_sq, local_mins, n_threads,
    )


# ``cache=True`` would require removing ``get_thread_id`` from the body, but
# that call is inside ``prange`` and cannot be hoisted — it resolves per
# iteration to the executing thread's ID. Omit caching on this kernel only.
@njit(parallel=True, fastmath=True, nogil=True)
def _njit_mp_stomp_kernel(
    X, m, scale, band_width, mins_sq, local_mins, n_threads,
):
    """Inner STOMP kernel. ``local_mins`` (n_threads, n) and ``n_threads``
    come from the caller so the per-thread reduction buffer is allocated
    before the parallel region."""
    sz = X.shape[0]
    d = X.shape[1]
    n = sz - m + 1
    INF = numpy.inf
    EPS = 1e-12

    for i in range(n):
        mins_sq[i] = INF
    # ``n <= 1`` is handled by the caller — kernel is only invoked
    # when there's at least one diagonal to walk, so n >= 2 here.

    # Sliding mean and std per-dim, used by both scaled formula and as
    # building blocks for the unscaled per-subsequence norm.
    mu = numpy.empty((n, d), dtype=numpy.float64)
    sig = numpy.empty((n, d), dtype=numpy.float64)
    for di in range(d):
        s = 0.0
        s2 = 0.0
        for k in range(m):
            v = X[k, di]
            s += v
            s2 += v * v
        mu[0, di] = s / m
        var = s2 / m - mu[0, di] * mu[0, di]
        if var < 0.0:
            var = 0.0
        sig[0, di] = numpy.sqrt(var)
        for i in range(1, n):
            v_in = X[i + m - 1, di]
            v_out = X[i - 1, di]
            s += v_in - v_out
            s2 += v_in * v_in - v_out * v_out
            mu[i, di] = s / m
            var = s2 / m - mu[i, di] * mu[i, di]
            if var < 0.0:
                var = 0.0
            sig[i, di] = numpy.sqrt(var)

    # Per-subsequence squared norm (only used when scale is False).
    norm_sq = numpy.empty(n, dtype=numpy.float64)
    if not scale:
        for i in range(n):
            norm_sq[i] = 0.0
        for di in range(d):
            s = 0.0
            for k in range(m):
                v = X[k, di]
                s += v * v
            norm_sq[0] += s
            for i in range(1, n):
                v_in = X[i + m - 1, di]
                v_out = X[i - 1, di]
                s += v_in * v_in - v_out * v_out
                norm_sq[i] += s

    # Thread-local mins to avoid races on diagonal updates: each diagonal
    # touches both endpoints of every cell along it, so two diagonals can
    # try to write the same ``mins_sq[i]``. ``local_mins`` and
    # ``n_threads`` come from the caller; ``get_thread_id`` still resolves
    # per-iteration inside ``prange`` so ``cache=True`` is not used.

    diag_lo = band_width + 1
    for diag in prange(diag_lo, n):
        tid = get_thread_id()
        # QT[i, i+diag] per dim; initial value is the dot product
        # ``sum_k X[k,di] * X[diag+k,di]``.
        qt = numpy.empty(d, dtype=numpy.float64)
        for di in range(d):
            q = 0.0
            for k in range(m):
                q += X[k, di] * X[diag + k, di]
            qt[di] = q

        max_i = n - diag
        for i in range(max_i):
            j = i + diag

            if i > 0:
                # STOMP recurrence: replace the leaving sample (i-1, j-1)
                # with the entering sample (i+m-1, j+m-1).
                for di in range(d):
                    qt[di] += (
                        X[i + m - 1, di] * X[j + m - 1, di]
                        - X[i - 1, di] * X[j - 1, di]
                    )

            if scale:
                dist_sq = 0.0
                cancellation_scale = 2.0 * m * d
                for di in range(d):
                    s_i = sig[i, di]
                    s_j = sig[j, di]
                    if s_i > EPS and s_j > EPS:
                        # The centered dot product and the variances lose
                        # precision in proportion to the squared mean/std
                        # ratios, even when the normalized distance is not
                        # tiny relative to 2*m alone.
                        cancellation_scale += 2.0 * m * (
                            (mu[i, di] / s_i) ** 2
                            + (mu[j, di] / s_j) ** 2
                        )
                        inner = (
                            qt[di] - m * mu[i, di] * mu[j, di]
                        ) / (s_i * s_j)
                        dist_sq += 2.0 * m - 2.0 * inner
                    elif s_i > EPS or s_j > EPS:
                        # One side is constant -> z is zero on that side;
                        # the other side has unit-variance z, so the per-dim
                        # contribution is ||z||^2 = m.
                        dist_sq += m
                    # else both constant: per-dim contribution is 0.
                # The correlation formula loses precision for nearly
                # identical normalized windows, just like the raw formula.
                # Leave headroom for roundoff accumulated by the recurrence.
                if dist_sq <= 1e-6 * cancellation_scale:
                    dist_sq = _scaled_distance_sq(X, i, j, m)
            else:
                qt_total = 0.0
                for di in range(d):
                    qt_total += qt[di]
                dist_sq = norm_sq[i] + norm_sq[j] - 2.0 * qt_total
                # Nearly equal, energetic windows lose their small distance
                # in the squared-norm subtraction. Recompute only these
                # pairs from differences, which do not suffer cancellation.
                if dist_sq <= 1e-6 * (norm_sq[i] + norm_sq[j]):
                    dist_sq = 0.0
                    for k in range(m):
                        for di in range(d):
                            diff = X[i + k, di] - X[j + k, di]
                            dist_sq += diff * diff

            if dist_sq < 0.0:
                dist_sq = 0.0

            if dist_sq < local_mins[tid, i]:
                local_mins[tid, i] = dist_sq
            if dist_sq < local_mins[tid, j]:
                local_mins[tid, j] = dist_sq

    for i in range(n):
        best = local_mins[0, i]
        for t in range(1, n_threads):
            v = local_mins[t, i]
            if v < best:
                best = v
        mins_sq[i] = best

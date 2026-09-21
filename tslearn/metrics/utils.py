import sys
import time
from contextlib import contextmanager

from joblib import Parallel, delayed, effective_n_jobs

from numba import (
    njit, config as _numba_config, get_num_threads, set_num_threads,
    threading_layer,
)

import numpy

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH
from tslearn.utils.utils import _to_time_series

from ._masks import GLOBAL_CONSTRAINT_CODE


_PROGRESS_TICKS = 50
# Each numba launch has fixed thread-pool / dispatch overhead in the low
# millisecond range. Below this many rows per chunk the launch cost
# starts to dominate the actual DP work, so cap the tick count by
# requiring chunks be at least this size.
_PROGRESS_MIN_CHUNK = 64


class _RowProgress:
    """In-place row progress for the chunked numba fast paths.

    Always writes to ``stderr`` so callers redirecting ``stdout`` for
    data don't get the heartbeat mixed in (mirrors tqdm / joblib).
    Off entirely when ``verbose`` is falsy or ``total`` is zero — the
    chunking caller still runs as a single chunk so ``verbose=0`` is
    free.
    """

    def __init__(self, total, verbose, label):
        self.total = int(total)
        self.enabled = bool(verbose) and self.total > 0
        self.label = label
        self.stream = sys.stderr
        self.start = time.perf_counter()
        self.last_len = 0

    def update(self, done):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        ratio = done / self.total
        rate = done / elapsed if elapsed > 0.0 else 0.0
        message = (
            f"\r{self.label}: {done}/{self.total} rows "
            f"({ratio:5.1%}) elapsed {elapsed:.1f}s"
        )
        if done < self.total and rate > 0.0:
            message += f" ETA {(self.total - done) / rate:.1f}s"
        elif done >= self.total:
            message += " done"
        padding = " " * max(0, self.last_len - len(message))
        self.stream.write(message + padding)
        self.stream.flush()
        self.last_len = len(message)

    def close(self):
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _row_chunk_size(n_rows, verbose):
    """Pick the per-launch chunk size for the row prange. ``verbose=0``
    keeps a single chunk so the kernel runs at full speed; otherwise
    target ``_PROGRESS_TICKS`` updates with a floor of
    ``_PROGRESS_MIN_CHUNK`` rows so per-launch overhead stays
    amortized on small inputs."""
    if not verbose:
        return n_rows
    target = (int(n_rows) + _PROGRESS_TICKS - 1) // _PROGRESS_TICKS
    return max(_PROGRESS_MIN_CHUNK, target)

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def _is_int_valued_finite(value) -> bool:
    """Whether ``value`` is a finite integer safe for band-kernel indices.

    Used to gate fast-path Sakoe-radius kernels (which need ``int`` for
    ``range(...)`` bounds) against public API inputs main accepts but
    can't be safely truncated — fractional radii change the mask, and
    NaN/Inf raise on ``int(...)``. Huge finite radii can also overflow
    Numba's int64 arguments or the subsequent band-bound additions.
    When this returns False, callers should defer to the legacy mask path.
    """
    f = float(value)
    # Keep headroom for adding series lengths to the signed int64 radius.
    return numpy.isfinite(f) and f.is_integer() and abs(f) < 2 ** 62


def _resolve_n_jobs(n_jobs):
    """Map sklearn-style ``n_jobs`` to a numba thread count, or ``None`` to
    leave numba's setting alone.

    - ``None`` or ``1`` -> 1 thread (sklearn convention).
    - ``-1`` -> ``None``: leave numba's setting untouched (respects
      ``NUMBA_NUM_THREADS`` and any prior ``set_num_threads`` call —
      we don't want to clobber an explicit user/env limit).
    - any other value is forwarded to ``joblib.effective_n_jobs``,
      which raises ``ValueError`` on ``0`` and resolves negative ``-N``
      (N>=2) to ``cpu_count() - (N - 1)`` for parity with the joblib
      ``Parallel`` calls used elsewhere in the codebase.
    """
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return None
    return effective_n_jobs(n_jobs)


def _assert_finite_over_valid_prefixes(X, lens=None):
    """Raise ``ValueError`` if any value within the valid (non-trailing-
    NaN) prefix of any row of ``X`` is NaN or +/-Inf.

    Trailing NaN is the variable-length sentinel — those positions are
    *not* checked. Mid-series NaN/Inf is the actual error case: the
    legacy (non-fast) paths raise via sklearn's ``assert_all_finite`` in
    ``SquaredEuclidean.compute`` or ``pairwise_euclidean_distances``,
    so the fast paths must agree.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, max_sz, d)
    lens : numpy.ndarray of shape (n,) or None
        Pre-computed valid lengths (``_compute_lengths`` output);
        recomputed on the fly when ``None``.
    """
    if X.shape[0] == 0 or X.shape[1] == 0:
        return
    if lens is None:
        lens = _compute_lengths(X)
    indices = numpy.arange(X.shape[1])
    valid_mask = indices[None, :] < lens[:, None]      # (n, max_sz)
    cell_finite = numpy.isfinite(X).all(axis=2)        # (n, max_sz)
    if not (cell_finite | ~valid_mask).all():
        raise ValueError(
            "Input contains NaN, infinity, or a value too large for "
            "dtype('float64')."
        )


def _raise_if_ambiguous_constraint(constraint_code, sakoe_chiba_radius,
                                    itakura_max_slope):
    """Raise ``RuntimeWarning`` if no global constraint is named but both
    ``sakoe_chiba_radius`` and ``itakura_max_slope`` are supplied — the
    same condition ``_compute_mask_generic`` raises on per-pair, lifted
    to a single check at the entry of cdist fast paths so they agree
    with the single-pair public API.

    ``constraint_code`` is the int code from ``GLOBAL_CONSTRAINT_CODE``;
    callers pre-normalize.
    """
    if (constraint_code == GLOBAL_CONSTRAINT_CODE[None]
            and sakoe_chiba_radius is not None
            and itakura_max_slope is not None):
        raise RuntimeWarning(
            "global_constraint is not set for DTW, but both "
            "sakoe_chiba_radius and itakura_max_slope are "
            "set, hence global_constraint cannot be inferred "
            "and no global constraint will be used."
        )


def _sakoe_radius_for_fast_path(metric_params):
    """Read the Sakoe-Chiba radius from ``metric_params`` if and only if
    the effective DTW constraint is Sakoe-Chiba (and the radius is
    non-negative, finite, and integer-valued). Otherwise return ``None``.

    Single entry point for any fast path that only knows how to do
    Sakoe-Chiba alignment (``cdist_dtw_topk_fast``, the band-iteration
    ``_njit_dtw_path_sakoe`` in DBA): callers don't need to remember to
    pass three keys to ``_is_sakoe_chiba_only`` at every site, and the
    ``radius >= 0`` guard lives in one place. Non-int-valued / non-finite
    radii also return ``None`` here so callers fall back to the mask
    path — main accepts these (the mask handles them) but the int-bound
    ``range(...)`` kernels these fast paths feed into can't.
    """
    # The shortcut consumes only these parameters. Let the regular caller
    # handle any others instead of silently ignoring them (including typos).
    if metric_params.keys() - {
        "global_constraint", "sakoe_chiba_radius", "itakura_max_slope",
    }:
        return None
    radius = metric_params.get("sakoe_chiba_radius")
    if radius is None or radius < 0:
        return None
    if not _is_int_valued_finite(radius):
        return None
    if not _is_sakoe_chiba_only(
        metric_params.get("global_constraint"),
        radius,
        metric_params.get("itakura_max_slope"),
    ):
        return None
    return radius


def _is_sakoe_chiba_only(global_constraint, sakoe_chiba_radius,
                         itakura_max_slope):
    """True iff the effective DTW constraint is Sakoe-Chiba and Itakura is
    fully absent.

    Used by fast paths that are Sakoe-Chiba-specific (the LB-prune top-k
    in neighbors/kmeans, the band-iteration ``_njit_dtw_path_sakoe`` in
    DBA) to decide whether to take the fast path or fall through.

    Mirrors the constraint logic of ``cdist_dtw_fast`` and
    ``_compute_mask_generic``. False on:

    - explicit ``global_constraint="itakura"`` (regardless of radius);
    - any path where ``itakura_max_slope`` is also set and Sakoe-Chiba
      isn't named explicitly — the surface-level ``cdist_dtw`` raises
      ``RuntimeWarning`` on this ambiguous combination, and the fast
      path must not silently disagree.

    ``global_constraint`` may be the public string form
    (``"itakura"`` / ``"sakoe_chiba"`` / ``""`` / ``None``) or the
    int-coded form from ``GLOBAL_CONSTRAINT_CODE``; both are normalized
    through that dict so a future code addition picks up here for free.
    """
    code = GLOBAL_CONSTRAINT_CODE.get(global_constraint, global_constraint)
    if code == GLOBAL_CONSTRAINT_CODE["itakura"]:
        return False
    if code == GLOBAL_CONSTRAINT_CODE["sakoe_chiba"]:
        return sakoe_chiba_radius is not None
    if code != GLOBAL_CONSTRAINT_CODE[None]:
        # Unknown names must reach the caller's regular validation path.
        return False
    # No explicit named constraint: only safe when a Sakoe radius is set
    # AND no Itakura slope is set (otherwise the combination is ambiguous
    # and the slow path raises).
    if sakoe_chiba_radius is None:
        return False
    return itakura_max_slope is None


def _numba_allows_concurrent_calls():
    """Whether parallel kernels can run from multiple Python threads."""
    # Public dispatchers use their legacy paths on workqueue so concurrent
    # callers cannot turn a regular API call into a process-wide abort.
    # Initialize before querying the actual selected layer, including when
    # Numba chooses workqueue automatically. Even one-thread launches abort
    # under workqueue if another Python thread is already using the pool.
    get_num_threads()
    return threading_layer() != "workqueue"


@contextmanager
def numba_threads_for(n_jobs):
    """Set numba's thread count for the duration of the ``with`` block,
    then restore the previous setting.

    Mapping (via ``_resolve_n_jobs``):

    - ``None`` or ``1`` -> set to 1 thread (sklearn convention).
    - positive int ``n`` -> set to ``n`` threads.
    - ``-1`` -> leave numba's thread count untouched (= all cores by
      default, or whatever the surrounding context has set).

    Usage::

        with numba_threads_for(n_jobs):
            _njit_kernel(...)
    """
    target = _resolve_n_jobs(n_jobs)
    if target is None:
        yield
        return
    # Numba's ``set_num_threads`` rejects requests above
    # ``NUMBA_NUM_THREADS`` (the configured pool size), but the legacy
    # joblib-backed API accepts oversubscribed positive ``n_jobs`` —
    # main returns a result for ``cdist_dtw(X, X, n_jobs=99)`` even
    # when only 12 cores are available. Clamp to keep that behavior.
    max_threads = _numba_config.NUMBA_NUM_THREADS
    if target > max_threads:
        target = max_threads
    prev = get_num_threads()
    set_num_threads(target)
    try:
        yield
    finally:
        set_num_threads(prev)


def _compute_lengths(X):
    """Vectorized batch counterpart to ``_ts_size``: returns ``(n,) int64`` of
    valid lengths (trailing all-NaN rows trimmed; mid-series NaN rows kept).
    """
    if X.shape[0] == 0 or X.shape[1] == 0:
        return numpy.zeros(X.shape[0], dtype=numpy.int64)
    not_all_nan = ~numpy.all(numpy.isnan(X), axis=2)
    indices = numpy.arange(X.shape[1])
    last_idx = numpy.where(not_all_nan, indices, -1).max(axis=1)
    return (last_idx + 1).astype(numpy.int64)


def _as_f64_contiguous(X):
    """``ascontiguousarray(X, float64)`` but skipping the copy when ``X`` is
    already C-contiguous float64 (the typical path from ``to_time_series_dataset``).
    """
    if X.dtype == numpy.float64 and X.flags.c_contiguous:
        return X
    return numpy.ascontiguousarray(X, dtype=numpy.float64)


def __make_compute_path(backend):

    def _compute_path_generic(acc_cost_mat):
        sz1, sz2 = acc_cost_mat.shape
        path = [(sz1 - 1, sz2 - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j - 1))
            elif j == 0:
                path.append((i - 1, 0))
            else:
                arr = backend.array(
                    [
                        acc_cost_mat[i - 1][j - 1],
                        acc_cost_mat[i - 1][j],
                        acc_cost_mat[i][j - 1],
                    ]
                )
                argmin = backend.argmin(arr)
                if argmin == 0:
                    path.append((i - 1, j - 1))
                elif argmin == 1:
                    path.append((i - 1, j))
                else:
                    path.append((i, j - 1))
        return path[::-1]
    if backend is numpy:
        return njit(nogil=True)(_compute_path_generic)
    else:
        return _compute_path_generic

_njit_compute_path = __make_compute_path(numpy)
if HAS_TORCH:
    _compute_path = __make_compute_path(instantiate_backend("torch"))
else:
    _compute_path = _njit_compute_path


def _cdist_generic(
    dist_fun,
    dataset1,
    dataset2,
    n_jobs,
    verbose,
    be,
    compute_diagonal=True,
    dtype=float,
    *args,
    **kwargs
):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.

    Parameters
    ----------
    dist_fun : function
        Similarity function to be used.

    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. 
        If `None`, self-similarity of `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.

    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.


    Returns
    -------
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.
    """  # noqa: E501
    n_ts_1 = len(dataset1)
    use_parallel = n_jobs not in [None, 1]

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = be.zeros((n_ts_1, n_ts_1), dtype=dtype)
        indices = be.triu_indices(
            n_ts_1, k=0 if compute_diagonal else 1, m=n_ts_1
        )

        if use_parallel:
            cdists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(dist_fun)(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset1[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(i if compute_diagonal else i + 1, n_ts_1)
            )
        else:
             cdists = [
                dist_fun(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset1[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(i if compute_diagonal else i + 1, n_ts_1)
            ]

        matrix[indices] = be.array(cdists, dtype=dtype)
        indices = be.tril_indices(n_ts_1, k=-1, m=n_ts_1)
        matrix[indices] = matrix.T[indices]

        return matrix
    else:
        n_ts_2 = len(dataset2)

        if use_parallel:
            cdists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(dist_fun)(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset2[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(n_ts_2)
            )
        else:
            cdists = [
                dist_fun(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset2[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(n_ts_2)
            ]
        matrix = be.array(cdists, dtype=dtype).reshape(n_ts_1, n_ts_2)
        return matrix

"""DTW Barycenter Averaging (DBA)"""
import warnings

from joblib import Parallel, delayed

import numpy

from scipy.interpolate import interp1d

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from tslearn.backend import instantiate_backend
from tslearn.barycenters.utils import _set_weights
from tslearn.metrics._dtw import _njit_dtw_path
from tslearn.metrics._dtw_fast import (
    _njit_dtw_path_sakoe,
    _njit_dtw_path_sakoe_arr,
)
from tslearn.metrics._masks import GLOBAL_CONSTRAINT_CODE
from tslearn.metrics.utils import _sakoe_radius_for_fast_path
from tslearn.utils import to_time_series_dataset, to_time_series
from tslearn.utils.utils import _to_time_series


def _dtw_path_dispatch(s1, s2, _path_as_array=False, **metric_params):
    """Route to the band-iteration fast path when ``metric_params`` selects
    Sakoe-Chiba (and not Itakura); otherwise use the legacy mask DP.
    ``_sakoe_radius_for_fast_path`` also rejects ambiguous params and
    explicit ``global_constraint="itakura"`` so callers don't silently
    get a Sakoe-Chiba alignment when they asked for something else.

    ``_path_as_array=True`` selects the variant that returns the path
    as a ``(path_len, 2)`` int64 ndarray — used internally by the DBA
    assignment loop, which would otherwise pay a per-call
    ``numpy.asarray(path, dtype=int64)`` to convert the numba list-of-
    tuples into a flat array.
    """
    radius = _sakoe_radius_for_fast_path(metric_params)
    if radius is not None:
        s1c = numpy.ascontiguousarray(s1, dtype=numpy.float64)
        s2c = numpy.ascontiguousarray(s2, dtype=numpy.float64)
        # The band-iteration path shortcut does not reproduce the legacy
        # mask DP's path choices when valid prefixes contain NaN/Inf. Fall
        # back so DBA assignments keep main's behavior on accepted inputs.
        if numpy.isfinite(s1c).all() and numpy.isfinite(s2c).all():
            if _path_as_array:
                return _njit_dtw_path_sakoe_arr(s1c, s2c, int(radius))
            return _njit_dtw_path_sakoe(s1c, s2c, int(radius))
    # ``_njit_dtw_path`` is the numba-jitted internal helper and expects
    # the int-coded constraint; the public string/None form silently
    # degrades to no constraint inside numba's type comparisons.
    # Normalize whenever the key is present (including explicit None,
    # a public value that maps to the int code 0); when absent, the
    # numba default kicks in. Don't mutate the caller's dict — DBA
    # reuses one ``metric_params`` across iterations and per-pair calls,
    # and a string→int swap on the first call would change later
    # ``_sakoe_radius_for_fast_path`` decisions.
    if "global_constraint" in metric_params:
        gc = metric_params["global_constraint"]
        metric_params = {
            **metric_params,
            "global_constraint": GLOBAL_CONSTRAINT_CODE.get(gc, gc),
        }
    dist, path = _njit_dtw_path(s1, s2, **metric_params)
    if _path_as_array:
        return dist, numpy.asarray(path, dtype=numpy.int64)
    return dist, path


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _init_avg(X, barycenter_size):
    if X.shape[1] == barycenter_size:
        return numpy.nanmean(X, axis=0)
    else:
        X_avg = numpy.nanmean(X, axis=0)
        xnew = numpy.linspace(0, 1, barycenter_size)
        f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)


def _petitjean_assignment(
    X,
    barycenter,
    weights,
    metric_params=None,
    n_jobs=None,
    trimmed=None,
):
    """Run DTW for every series against the current ``barycenter`` and
    return the assignment as three flat arrays of equal length.

    Each row ``k`` represents one (series, time-in-series, time-in-
    barycenter) triple from some series' DTW path:

    - ``i_arr[k]``: series index (``0..n``)
    - ``t_arr[k]``: barycenter position (``0..barycenter_size``)
    - ``j_arr[k]``: time index inside ``X[i_arr[k]]``

    Flat arrays let ``_petitjean_update_barycenter`` do one
    ``numpy.add.at`` + one ``numpy.bincount`` per iteration instead of
    ``barycenter_size`` ``numpy.average`` calls — the per-call Python
    overhead inside ``numpy.average`` was the dominant DBA cost on small
    inputs.

    ``trimmed`` is an optional pre-computed list of ``_to_time_series``
    outputs (one per series) so the per-pair NaN-trim work is hoisted
    out of the inner loop. Fall back to per-call trimming when ``None``.
    """
    backend = instantiate_backend("numpy")
    use_parallel = n_jobs not in [None, 1]

    if metric_params is None:
        metric_params = {}
    n = X.shape[0]

    def _trimmed_at(i):
        return trimmed[i] if trimmed is not None else _to_time_series(
            X[i], True, backend,
        )

    # ``_path_as_array=True`` makes the dispatcher return paths as
    # ``(path_len, 2)`` int64 arrays (kernels build them directly;
    # legacy fallback does one ``numpy.asarray``). Either way, we don't
    # iterate per-tuple in Python, which was ~50ms/run on small inputs.
    if use_parallel:
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_dtw_path_dispatch)(
                _trimmed_at(i),
                barycenter,
                _path_as_array=True,
                **metric_params
            )
            for i in range(n)
        )
    else:
        res = [
            _dtw_path_dispatch(
                _trimmed_at(i), barycenter,
                _path_as_array=True, **metric_params,
            )
            for i in range(n)
        ]

    i_chunks = []
    t_chunks = []
    j_chunks = []
    cost = 0.0
    for i, (dist_i, path_arr) in enumerate(res):
        cost += dist_i ** 2 * weights[i]
        i_chunks.append(numpy.full(path_arr.shape[0], i, dtype=numpy.int64))
        j_chunks.append(path_arr[:, 0])
        t_chunks.append(path_arr[:, 1])

    cost /= weights.sum()
    if i_chunks:
        i_arr = numpy.concatenate(i_chunks)
        t_arr = numpy.concatenate(t_chunks)
        j_arr = numpy.concatenate(j_chunks)
    else:
        i_arr = numpy.empty(0, dtype=numpy.int64)
        t_arr = numpy.empty(0, dtype=numpy.int64)
        j_arr = numpy.empty(0, dtype=numpy.int64)
    return (i_arr, t_arr, j_arr), cost


def _petitjean_update_barycenter(X, assign, barycenter_size, weights):
    """Vectorized barycenter update: scatter weighted contributions into
    the new barycenter via ``numpy.add.at`` + ``numpy.bincount`` instead
    of a per-position ``numpy.average`` loop. Equivalent up to float
    reduction order.

    ``assign`` is the ``(i_arr, t_arr, j_arr)`` triple from
    ``_petitjean_assignment`` — one row per (series, barycenter pos,
    time-in-series) triple along every series' DTW path.
    """
    i_arr, t_arr, j_arr = assign
    d = X.shape[-1]
    bary_sum = numpy.zeros((barycenter_size, d), dtype=numpy.float64)
    if i_arr.size == 0:
        # No path triples — every t-bin has zero weight, so main's
        # per-position ``numpy.average`` would raise on the first one.
        raise ZeroDivisionError(
            "Weights sum to zero, can't be normalized"
        )
    w_arr = weights[i_arr]
    contrib = X[i_arr, j_arr] * w_arr[:, None]
    numpy.add.at(bary_sum, t_arr, contrib)
    bary_w = numpy.bincount(t_arr, weights=w_arr, minlength=barycenter_size)
    # Match the per-position ``numpy.average`` raise on main: any t-bin
    # whose weight sum is 0 (typically all input weights are 0, but
    # also reachable when only zero-weighted series visit position t)
    # bubbled up as ``ZeroDivisionError: Weights sum to zero, can't be
    # normalized``. The vectorized division would silently emit NaN.
    if (bary_w == 0).any():
        raise ZeroDivisionError(
            "Weights sum to zero, can't be normalized"
        )
    return bary_sum / bary_w[:, None]


def dtw_barycenter_averaging_petitjean(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_jobs=None
):
    """DTW Barycenter Averaging (DBA) method.

    DBA was originally presented in [1]_.
    This implementation is not the one documented in the API, but is kept
    in the codebase to check the documented one for non-regression.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging_petitjean(time_series, max_iter=5,
    ...                                    barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging_petitjean([[0, 0, 0], [10, 10, 10]],
    ...                                    weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    backend = instantiate_backend("numpy")

    X_ = to_time_series_dataset(X, be=backend)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = to_time_series(
            init_barycenter,
            remove_nans=True,
            be=backend
        )
    # ``_to_time_series`` trims trailing NaN rows; the result is identical
    # for every DBA iteration so cache it once instead of paying the
    # NaN-scan overhead per (iteration, series) pair.
    trimmed = [_to_time_series(X_[i], True, backend) for i in range(X_.shape[0])]
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        assign, cost = _petitjean_assignment(
            X_, barycenter, weights, metric_params, n_jobs, trimmed=trimmed,
        )
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _petitjean_update_barycenter(X_, assign, barycenter_size,
                                                  weights)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter


def _mm_assignment(
    X,
    barycenter,
    weights,
    metric_params=None,
    n_jobs=None
):
    """Computes item assignment based on DTW alignments and returns cost as a
    bonus.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the current step of the algorithm.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    Returns
    -------
    list of index pairs
        Warping paths

    float
        Current alignment cost
    """
    backend = instantiate_backend("numpy")
    use_parallel = n_jobs not in [None, 1]

    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    cost = 0.
    list_p_k = []

    if use_parallel:
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_dtw_path_dispatch)(
                barycenter,
                _to_time_series(X[i], True, backend),
                **metric_params
            )
            for i in range(n)
        )

        for i, (dist, path) in enumerate(res):
            cost += dist ** 2 * weights[i]
            list_p_k.append(path)

    else:
        for i in range(n):
            dist_i, path = _dtw_path_dispatch(
                barycenter,
                _to_time_series(X[i], True, backend),
                **metric_params
            )
            cost += dist_i ** 2 * weights[i]
            list_p_k.append(path)

    cost /= weights.sum()
    return list_p_k, cost


def _subgradient_valence_warping(list_p_k, barycenter_size, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns a list of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k = []
    list_w_k = []
    for k, p_k in enumerate(list_p_k):
        sz_k = p_k[-1][1] + 1
        w_k = numpy.zeros((barycenter_size, sz_k))
        for i, j in p_k:
            w_k[i, j] = 1.
        list_w_k.append(w_k * weights[k])
        list_v_k.append(w_k.sum(axis=1) * weights[k])
    return list_v_k, list_w_k


def _mm_valence_warping(list_p_k, barycenter_size, weights):
    """Compute Valence and Warping matrices from paths.

    Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
    :math:`W^{(k)}` in [1]_.

    This function returns the sum of :math:`V^{(k)}` diagonals (as a vector)
    and a list of :math:`W^{(k)}` matrices.

    Parameters
    ----------
    list_p_k : list of index pairs
        Warping paths

    barycenter_size : int
        Size of the barycenter to generate.

    weights: array
        Weights of each X[i]. Must be the same size as len(X).

    Returns
    -------
    numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    list_v_k, list_w_k = _subgradient_valence_warping(
        list_p_k=list_p_k,
        barycenter_size=barycenter_size,
        weights=weights)
    diag_sum_v_k = numpy.zeros(list_v_k[0].shape)
    for v_k in list_v_k:
        diag_sum_v_k += v_k
    return diag_sum_v_k, list_w_k


def _mm_update_barycenter(X, diag_sum_v_k, list_w_k):
    """Update barycenters using the formula from Algorithm 2 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    diag_sum_v_k : numpy.array of shape (barycenter_size, )
        sum of weighted :math:`V^{(k)}` diagonals (as a vector)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    backend = instantiate_backend("numpy")

    d = X.shape[2]
    barycenter_size = diag_sum_v_k.shape[0]
    n = len(list_w_k)
    # When all w_k share the same shape (the common equal-length case),
    # stack and use a single batched matmul instead of a Python loop. The
    # candidate widths come from path lengths and match each X[i]'s valid
    # length; for equal-length input, that's a uniform sz.
    sz_first = list_w_k[0].shape[1] if n else 0
    same_shape = n > 0 and all(w.shape[1] == sz_first for w in list_w_k)
    if same_shape and n > 0:
        W = numpy.stack(list_w_k)  # (n, barycenter_size, sz)
        # Build (n, sz, d) of trimmed series. _to_time_series trims trailing
        # NaN rows; equal-length & same-shape implies sz_first matches the
        # trimmed length here.
        Xt = numpy.empty((n, sz_first, d))
        for k in range(n):
            Xt[k] = _to_time_series(X[k], True, backend)
        sum_w_x = numpy.einsum("kij,kjd->id", W, Xt)
    else:
        sum_w_x = numpy.zeros((barycenter_size, d))
        for k in range(n):
            sum_w_x += list_w_k[k].dot(
                _to_time_series(X[k], True, backend)
            )
    return sum_w_x / diag_sum_v_k[:, None]


def _subgradient_update_barycenter(X, list_diag_v_k, list_w_k, weights_sum,
                                   barycenter, eta):
    """Update barycenters using the formula from Algorithm 1 in [1]_.

    Parameters
    ----------
    X : numpy.array of shape (n, sz, d)
        Time-series to be averaged

    list_diag_v_k : list of numpy.array of shape (barycenter_size, )
        list of weighted :math:`V^{(k)}` diagonals (as vectors)

    list_w_k : list of numpy.array of shape (barycenter_size, sz_k)
        list of weighted :math:`W^{(k)}` matrices

    weights_sum : float
        sum of weights applied to matrices :math:`V^{(k)}` and :math:`W^{(k)}`

    barycenter : numpy.array of shape (barycenter_size, d)
        Barycenter as computed at the previous iteration of the algorithm

    eta : float
        Step-size for the subgradient descent algorithm

    Returns
    -------
    numpy.array of shape (barycenter_size, d)
        Updated barycenter

    References
    ----------

    .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    backend = instantiate_backend("numpy")

    d = X.shape[2]
    barycenter_size = barycenter.shape[0]
    delta_bar = numpy.zeros((barycenter_size, d))
    for k, (v_k, w_k, x_k) in enumerate(zip(list_diag_v_k, list_w_k, X)):
        delta_bar += v_k.reshape((-1, 1)) * barycenter
        delta_bar -= w_k.dot(_to_time_series(x_k, True, backend))
    barycenter -= (2. * eta / weights_sum) * delta_bar
    return barycenter


def dtw_barycenter_averaging(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_init=1,
    n_jobs=None
):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on an idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_init : int (default: 1)
        Number of different initializations to be tried (useful only is
        init_barycenter is set to None, otherwise, all trials will reach the
        same performance)

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5,
    ...                          metric_params={"itakura_max_slope": 2})
    array([[1. ],
           [2. ],
           [3. ],
           [3.5],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5, barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])
    >>> dtw_barycenter_averaging([[0, 0, 0], [10, 10, 10]], max_iter=1,
    ...                          weights=numpy.array([0.75, 0.25]))
    array([[2.5],
           [2.5],
           [2.5]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    best_cost = numpy.inf
    best_barycenter = None
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
        bary, loss = dtw_barycenter_averaging_one_init(
            X=X,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            verbose=verbose,
            n_jobs=n_jobs
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
    return best_barycenter


def dtw_barycenter_averaging_one_init(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_jobs=None
):
    """DTW Barycenter Averaging (DBA) method estimated through
    Expectation-Maximization algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Majorize-Minimize Mean
    Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.


    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.
    float
        Associated inertia

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    backend = instantiate_backend("numpy")

    X_ = to_time_series_dataset(X, be=backend)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = to_time_series(
            init_barycenter,
            remove_nans=True,
            be=backend
        )
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        list_p_k, cost = _mm_assignment(
            X_,
            barycenter,
            weights,
            metric_params,
            n_jobs
        )
        diag_sum_v_k, list_w_k = _mm_valence_warping(list_p_k, barycenter_size,
                                                     weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _mm_update_barycenter(X_, diag_sum_v_k, list_w_k)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter, cost


def dtw_barycenter_averaging_subgradient(
    X,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    initial_step_size=.05,
    final_step_size=.005,
    tol=1e-5,
    random_state=None,
    weights=None,
    metric_params=None,
    verbose=False
):
    """DTW Barycenter Averaging (DBA) method estimated through subgradient
    descent algorithm.

    DBA was originally presented in [1]_.
    This implementation is based on a idea from [2]_ (Stochastic Subgradient
    Mean Algorithm).

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    initial_step_size : float (default: 0.05)
        Initial step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    final_step_size : float (default: 0.005)
        Final step size for the subgradient descent algorithm.
        Default value is the one suggested in [2]_.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging_subgradient(
    ...     time_series,
    ...     max_iter=10,
    ...     random_state=0
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1. ],
           [2. ],
           [3.5...],
           [4.5...]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """
    rng = check_random_state(random_state)
    backend = instantiate_backend("numpy")

    X_ = to_time_series_dataset(X, be=backend)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter = to_time_series(
            init_barycenter,
            remove_nans=True,
            be=backend
        )
        barycenter_size = barycenter.shape[0]
    cost_prev, cost = numpy.inf, numpy.inf
    eta = initial_step_size
    n = X_.shape[0]
    for it in range(max_iter):
        shuffled_indices = rng.permutation(n)
        for idx in shuffled_indices:
            Xi = X_[idx:idx+1]
            wi = weights[idx:idx+1]
            list_p_k, cost = _mm_assignment(Xi, barycenter, weights,
                                            metric_params)
            list_diag_v_k, list_w_k = _subgradient_valence_warping(
                list_p_k,
                barycenter_size,
                wi
            )
            if verbose:
                print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            barycenter = _subgradient_update_barycenter(Xi, list_diag_v_k,
                                                        list_w_k, wi.sum(),
                                                        barycenter, eta)
            if it == 0:
                eta -= (initial_step_size - final_step_size) / n
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter

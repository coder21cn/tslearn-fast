# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

import numpy

from scipy.optimize import minimize

from tslearn.backend import instantiate_backend
from tslearn.metrics import SquaredEuclidean, SoftDTW
from tslearn.metrics._softdtw_fast import softdtw_obj_grad_fast
from tslearn.metrics.utils import _numba_allows_concurrent_calls
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from tslearn.utils.utils import _check_equal_size, _to_time_series

from .utils import _set_weights
from .euclidean import euclidean_barycenter

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _softdtw_func(Z, X, weights, barycenter, gamma):
    # Compute objective value and grad at Z.

    Z = Z.reshape(barycenter.shape)
    G = numpy.zeros_like(Z)
    obj = 0

    for i in range(len(X)):
        D = SquaredEuclidean(Z, X[i])
        sdtw = SoftDTW(D, gamma=gamma)
        value = sdtw.compute()
        E = sdtw.grad()
        G_tmp = D.jacobian_product(E)
        G += weights[i] * G_tmp
        obj += weights[i] * value

    return obj, G.ravel()


def softdtw_barycenter(X, gamma=1.0, weights=None, method="L-BFGS-B", tol=1e-3,
                       max_iter=50, init=None):
    """Compute barycenter (time series averaging) under the soft-DTW
    geometry.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.
    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    tol: float
        Tolerance of the method used.
    max_iter: int
        Maximum number of iterations.
    init: array or None (default: None)
        Initial barycenter to start from for the optimization process.
        If `None`, euclidean barycenter is used as a starting point.

    Returns
    -------
    numpy.array of shape (bsz, d) where `bsz` is the size of the `init` array \
            if provided or `sz` otherwise
        Soft-DTW barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> softdtw_barycenter(time_series, max_iter=5)
    array([[1.25161574],
           [2.03821705],
           [3.5101956 ],
           [4.36140605]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> softdtw_barycenter(time_series, max_iter=5)
    array([[1.21349933],
           [1.8932251 ],
           [2.67573269],
           [3.51057026],
           [4.33645802]])

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    backend = instantiate_backend(X)
    X_ = to_time_series_dataset(X, be=backend)

    weights = _set_weights(weights, X_.shape[0])
    if init is None:
        if _check_equal_size(X_):
            barycenter = euclidean_barycenter(X_, weights)
        else:
            resampled_X = TimeSeriesResampler(sz=X_.shape[1]).fit_transform(X_)
            barycenter = euclidean_barycenter(resampled_X, weights)
    else:
        barycenter = init

    if max_iter > 0:
        # Reject zero-length / fully-NaN series. The fused per-i objective
        # contributes 0/0 for these (matching the legacy `_softdtw_func`
        # behavior), which silently drops them from the barycenter
        # optimization. Soft-DTW barycenter against empty series is
        # mathematically undefined; raise so the caller fixes their input.
        # Gated on max_iter > 0 because a 0-iter call returns init
        # unchanged on main without inspecting members.
        if backend.is_numpy and X_.shape[0] > 0:
            from tslearn.utils.utils import _ts_size
            for i in range(X_.shape[0]):
                if _ts_size(X_[i]) == 0:
                    raise ValueError(
                        "softdtw_barycenter input contains a zero-length / "
                        "all-NaN time series at index {}; the barycenter "
                        "optimization is undefined for empty members.".format(i)
                    )

        # Numpy fast path: pre-pad the per-series tensor once; the fused
        # parallel kernel computes (value, G) for all i in parallel inside
        # each scipy.optimize callback. Zero / non-finite gamma falls through
        # to the legacy ``_softdtw_func`` so divide-by-gamma surfaces the
        # same exception main raises.
        if (backend.is_numpy and gamma != 0.0 and numpy.isfinite(gamma)
                and _numba_allows_concurrent_calls()):
            from tslearn.metrics.utils import (
                _assert_finite_over_valid_prefixes,
            )
            from sklearn.utils.validation import assert_all_finite
            trimmed = [_to_time_series(d, True, backend) for d in X_]
            n_X = len(trimmed)
            d = barycenter.shape[1]
            if d != X_.shape[2]:
                raise ValueError(
                    "Incompatible dimension for X and Y matrices: "
                    f"X.shape[1] == {d} while Y.shape[1] == {X_.shape[2]}"
                )
            max_n = max((t.shape[0] for t in trimmed), default=0)
            X_padded = numpy.zeros((n_X, max_n, d), dtype=numpy.float64)
            lens_X = numpy.empty(n_X, dtype=numpy.int64)
            for i, t in enumerate(trimmed):
                lens_X[i] = t.shape[0]
                X_padded[i, :t.shape[0], :] = t
            # The fused per-i objective doesn't go through SquaredEuclidean,
            # so mid-series NaN/Inf in the inputs (or the init/barycenter)
            # would silently produce a 0-objective + 0-gradient and the
            # L-BFGS would return the initial barycenter. The legacy
            # `_softdtw_func` path raises via SquaredEuclidean.compute.
            _assert_finite_over_valid_prefixes(X_padded, lens=lens_X)
            assert_all_finite(barycenter)
            w64 = numpy.asarray(weights, dtype=numpy.float64)
            bary_shape = barycenter.shape

            def f(Z):
                # The optimizer can propose NaN/Inf even from finite init.
                # The legacy SquaredEuclidean path validates every iterate.
                assert_all_finite(Z)
                obj, G = softdtw_obj_grad_fast(
                    Z.reshape(bary_shape), X_padded, lens_X, w64, gamma
                )
                return obj, G.ravel()
        else:
            X_ = [_to_time_series(d, True, backend) for d in X_]

            def f(Z):
                return _softdtw_func(Z, X_, weights, barycenter, gamma)

        # The function works with vectors so we need to vectorize barycenter.
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                       options=dict(maxiter=max_iter))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter

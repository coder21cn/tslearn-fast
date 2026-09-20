"""Subprocess entrypoint for the differential parity harness.

Loads a manifest of ``(case_id, dotted_func, args, kwargs)`` tuples from
``argv[1]``, runs each call, and writes a list of ``(case_id, result)``
tuples to ``argv[2]``. Each result is one of:

    {"status": "ok",  "result": <pickleable>}
    {"status": "err", "type": <exc cls name>, "msg": <truncated str>}

Run under whichever ``PYTHONPATH`` selects the tslearn install you want
to probe (e.g. a ``main`` worktree).
"""
import importlib
import numpy as np
import pickle
import sys
import warnings


LOCAL_PREFIX = "local:"


def run_resampler(ds, sz):
    """Resample ``ds`` to length ``sz`` via ``TimeSeriesResampler``."""
    from tslearn.preprocessing import TimeSeriesResampler
    return TimeSeriesResampler(sz=sz).fit_transform(ds)


def run_scaler_minmax(ds, value_range, per_ts, per_feat):
    """``TimeSeriesScalerMinMax.fit_transform`` end-to-end."""
    from tslearn.preprocessing import TimeSeriesScalerMinMax
    return TimeSeriesScalerMinMax(
        value_range=value_range,
        per_timeseries=per_ts,
        per_feature=per_feat,
    ).fit_transform(ds)


def run_scaler_meanvar(ds, mu, std, per_ts, per_feat):
    """``TimeSeriesScalerMeanVariance.fit_transform`` end-to-end."""
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    return TimeSeriesScalerMeanVariance(
        mu=mu, std=std, per_timeseries=per_ts, per_feature=per_feat,
    ).fit_transform(ds)


def run_paa(ds, n_segments):
    """``PiecewiseAggregateApproximation.fit_transform`` end-to-end."""
    from tslearn.piecewise import PiecewiseAggregateApproximation
    return PiecewiseAggregateApproximation(
        n_segments=n_segments,
    ).fit_transform(ds)


def run_sax(ds, n_segments, alphabet_size_avg, scale):
    """``SymbolicAggregateApproximation.fit_transform`` end-to-end."""
    from tslearn.piecewise import SymbolicAggregateApproximation
    return SymbolicAggregateApproximation(
        n_segments=n_segments,
        alphabet_size_avg=alphabet_size_avg,
        scale=scale,
    ).fit_transform(ds)


def run_imputer(ds, method, value, keep_trailing):
    """``TimeSeriesImputer.fit_transform`` end-to-end. Exercises the
    NaN-handling paths across the supported imputation strategies."""
    from tslearn.preprocessing import TimeSeriesImputer
    return TimeSeriesImputer(
        method=method, value=value, keep_trailing_nans=keep_trailing,
    ).fit_transform(ds)


def run_svr_gak_predict(X_train, y_train, X_test, gamma):
    """End-to-end ``TimeSeriesSVR(kernel='gak').predict``.

    Regression sibling of ``run_svc_gak_predict``; exercises the same
    GAK self-diagonal cache and predict-time normalization re-use,
    against a continuous target rather than class labels.
    """
    from tslearn.svm import TimeSeriesSVR
    svr = TimeSeriesSVR(kernel="gak", gamma=gamma)
    svr.fit(X_train, y_train)
    return svr.predict(X_test)


def _result_to_numpy(obj):
    """Recursively convert ``torch.Tensor`` results to ``numpy.ndarray``
    so the harness comparator (numpy-only) sees them as ndarrays.
    Tuples / lists recurse; everything else passes through."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, tuple):
        return tuple(_result_to_numpy(x) for x in obj)
    if isinstance(obj, list):
        return [_result_to_numpy(x) for x in obj]
    return obj


def run_torch_dispatch(dotted_func, args, kwargs):
    """Call ``dotted_func(*args, **kwargs)`` after promoting every
    ``numpy.ndarray`` in ``args`` to a ``torch.float64`` tensor, then
    coerce the result tree back to numpy. Exercises the ``be.is_numpy``
    branch in the metric dispatchers — any code path that takes the
    fast numba route on a torch input is a parity bug.
    """
    import torch
    mod_path, _, attr = dotted_func.rpartition(".")
    mod = importlib.import_module(mod_path)
    func = getattr(mod, attr)

    def _to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(
                np.ascontiguousarray(x, dtype=np.float64)
            )
        return x

    promoted = tuple(_to_torch(a) for a in args)
    return _result_to_numpy(func(*promoted, **kwargs))


def run_matrix_profile(ds, sub_len, scale):
    """Compute the numpy-backend matrix profile of ``ds``."""
    from tslearn.matrix_profile import MatrixProfile
    return MatrixProfile(
        subsequence_length=sub_len, implementation="numpy", scale=scale,
    ).fit_transform(ds)


def run_kneighbors_dtw(X_train, X_test, n_neighbors, radius):
    """End-to-end ``KNeighborsTimeSeries(metric='dtw').kneighbors``.

    Returns ``(distances_sorted_per_row, sorted_indices_per_row)``.
    Index ties on equal distances aren't deterministic across the LB
    fast path and the legacy cdist path, so callers should assert on
    sorted distances rather than raw indices. ``radius`` is forwarded
    as-is — fractional / non-finite values exercise the
    ``_sakoe_radius_for_fast_path`` fallback gate.
    """
    from tslearn.neighbors import KNeighborsTimeSeries
    knn = KNeighborsTimeSeries(
        n_neighbors=n_neighbors,
        metric="dtw",
        metric_params={"sakoe_chiba_radius": radius},
    ).fit(X_train)
    dists, idx = knn.kneighbors(X_test, return_distance=True)
    return dists, idx


def run_svc_gak_predict(X_train, y_train, X_test, gamma):
    """End-to-end ``TimeSeriesSVC(kernel='gak').predict``.

    Trains on ``(X_train, y_train)`` then returns predicted labels for
    ``X_test``. Exercises the fit-time GAK self-diagonal cache + the
    predict-time normalization re-use that the perf branch added.
    """
    from tslearn.svm import TimeSeriesSVC
    svc = TimeSeriesSVC(kernel="gak", gamma=gamma, random_state=0)
    svc.fit(X_train, y_train)
    return svc.predict(X_test)


def run_softdtw_barycenter(ds, gamma, max_iter, init=None):
    """End-to-end ``softdtw_barycenter`` — runs the full L-BFGS loop."""
    from tslearn.barycenters import softdtw_barycenter
    return softdtw_barycenter(
        ds, gamma=gamma, max_iter=max_iter, init=init,
    )


def _cluster_summary(est):
    """Compact, pickleable estimator state for parity comparisons."""
    out = [np.asarray(est.labels_), float(est.inertia_)]
    if hasattr(est, "cluster_centers_") and est.cluster_centers_ is not None:
        out.append(np.asarray(est.cluster_centers_))
    if hasattr(est, "n_iter_"):
        out.append(int(est.n_iter_))
    return tuple(out)


def run_timeseries_kmeans(ds, metric, metric_params, n_clusters=2):
    """Fit ``TimeSeriesKMeans`` and return stable comparable state.

    Covers the DTW and Soft-DTW estimator surface using deterministic
    random initialization. ``metric_params`` is copied so the estimator can
    mutate it without affecting subsequent cases in this subprocess.
    """
    from tslearn.clustering import TimeSeriesKMeans
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=metric,
        metric_params=dict(metric_params or {}),
        max_iter=2,
        max_iter_barycenter=2,
        n_init=1,
        random_state=0,
        init="random",
        verbose=False,
    )
    km.fit(ds)
    pred = km.predict(ds)
    return _cluster_summary(km) + (np.asarray(pred),)


def run_kernel_kmeans_gak(ds, sigma, n_clusters=2):
    """Fit ``KernelKMeans(kernel='gak')`` and return comparable state."""
    from tslearn.clustering import KernelKMeans
    km = KernelKMeans(
        n_clusters=n_clusters,
        kernel="gak",
        kernel_params={"sigma": sigma},
        max_iter=2,
        n_init=1,
        random_state=0,
        verbose=False,
    )
    km.fit(ds)
    pred = km.predict(ds)
    return _cluster_summary(km) + (np.asarray(pred),)


def run_kshape(ds, n_clusters=2):
    """Fit ``KShape`` and return stable comparable state."""
    from tslearn.clustering import KShape
    ks = KShape(
        n_clusters=n_clusters,
        max_iter=2,
        n_init=1,
        random_state=0,
        init="random",
        verbose=False,
    )
    ks.fit(ds)
    return _cluster_summary(ks)


def _resolve(dotted):
    """Resolve ``dotted`` to a callable.

    ``local:<name>`` looks up ``<name>`` in this module's globals; any
    other dotted name is imported via ``importlib``. The ``local:``
    indirection lets the runner reach helpers that the ``main`` worktree
    wouldn't have a way to import (the runner script is only on HEAD).
    """
    if dotted.startswith(LOCAL_PREFIX):
        return globals()[dotted[len(LOCAL_PREFIX):]]
    mod_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def run_one(func_dotted, args, kwargs):
    """Execute ``func_dotted(*args, **kwargs)`` and capture the outcome.

    Returns ``{"status": "ok", "result": ...}`` or
    ``{"status": "err", "type": <cls name>, "msg": <truncated str>}``.
    Imported by both the subprocess entry below and ``run_on_head`` in
    the harness driver, so the result-shape contract has one source.
    """
    try:
        func = _resolve(func_dotted)
    except Exception as e:
        return {"status": "err", "type": type(e).__name__,
                "msg": "import: " + str(e)[:200]}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = func(*args, **kwargs)
    except Exception as e:
        return {"status": "err", "type": type(e).__name__,
                "msg": str(e)[:300]}
    return {"status": "ok", "result": result}


def main(in_path, out_path):
    with open(in_path, "rb") as f:
        cases = pickle.load(f)
    out = [(cid, run_one(fn, args, kw)) for cid, fn, args, kw in cases]
    with open(out_path, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

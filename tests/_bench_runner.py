"""Subprocess entrypoint for ``bench_vs_main.py``.

Reads ``(scenarios, repeats)`` from argv[1], runs each scenario
``repeats`` times after a single warm-up call (to amortize JIT compile
cost), and writes ``{case_id: median_seconds}`` to argv[2].
"""
import importlib
import pickle
import statistics
import sys
import time


def run_kneighbors_dtw(X_train, X_test, n_neighbors, radius):
    from tslearn.neighbors import KNeighborsTimeSeries
    knn = KNeighborsTimeSeries(
        n_neighbors=n_neighbors,
        metric="dtw",
        metric_params={"sakoe_chiba_radius": int(radius)},
    ).fit(X_train)
    return knn.kneighbors(X_test, return_distance=True)


def run_kmeans_dtw(X, n_clusters, radius):
    from tslearn.clustering import TimeSeriesKMeans
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        metric_params={"sakoe_chiba_radius": int(radius)},
        max_iter=3,
        n_init=1,
        random_state=0,
    )
    return km.fit_predict(X)


def run_softdtw_barycenter(ds, gamma, max_iter):
    from tslearn.barycenters import softdtw_barycenter
    return softdtw_barycenter(ds, gamma=gamma, max_iter=max_iter)


def run_matrix_profile(ds, sub_len, scale):
    from tslearn.matrix_profile import MatrixProfile
    return MatrixProfile(
        subsequence_length=sub_len, implementation="numpy", scale=scale,
    ).fit_transform(ds)


def _resolve(dotted):
    if dotted.startswith("local:"):
        return globals()[dotted[len("local:"):]]
    mod_path, _, attr = dotted.rpartition(".")
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _time_one(func, args, kwargs, repeats):
    # Warm-up: pays JIT compile cost so it's not counted in the
    # measurement. Each subprocess JITs once even with cache=True
    # (cache hits still parse the cache file).
    func(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main(in_path, out_path):
    with open(in_path, "rb") as f:
        scenarios, repeats = pickle.load(f)
    out = {}
    for case_id, func_dotted, args, kwargs in scenarios:
        func = _resolve(func_dotted)
        out[case_id] = _time_one(func, args, kwargs, repeats)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""End-to-end clustering benchmarks: KMeans (DTW + SoftDTW) and KShape.

These exercise the outer-parallel barycenter loops added in the perf work
plus the LB top-k assignment fast path (KMeans+DTW).
"""
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from ._common import make_dataset, run_standalone


class KMeansDTW:
    params = (
        [60, 120],   # n_ts
        [64, 200],   # sz
        [4, 8],      # n_clusters
    )
    param_names = ["n_ts", "sz", "n_clusters"]

    def setup(self, n_ts, sz, n_clusters):
        self.X = make_dataset(n_ts, sz, d=1)
        self.n_clusters = n_clusters

    def time_fit(self, n_ts, sz, n_clusters):
        TimeSeriesKMeans(
            n_clusters=n_clusters, metric="dtw", n_init=1, max_iter=5,
            random_state=0, n_jobs=-1,
            metric_params={"sakoe_chiba_radius": max(2, sz // 20)},
        ).fit(self.X)


class KMeansSoftDTW:
    params = (
        [60, 120],   # n_ts
        [64, 128],   # sz
        [4, 8],      # n_clusters
    )
    param_names = ["n_ts", "sz", "n_clusters"]

    def setup(self, n_ts, sz, n_clusters):
        self.X = make_dataset(n_ts, sz, d=1)
        self.n_clusters = n_clusters

    def time_fit(self, n_ts, sz, n_clusters):
        TimeSeriesKMeans(
            n_clusters=n_clusters, metric="softdtw",
            metric_params={"gamma": 1.0},
            n_init=1, max_iter=5, max_iter_barycenter=5,
            random_state=0, n_jobs=-1,
        ).fit(self.X)


class KShapeFit:
    # Univariate only — KShape's hot path (cross-correlation + eigh) does
    # not change shape with `d` the way DTW kernels do, so adding a `d`
    # axis here would double cell count without measuring anything new.
    params = (
        [60, 120],   # n_ts
        [128, 256],  # sz
        [4, 8],      # n_clusters
    )
    param_names = ["n_ts", "sz", "n_clusters"]

    def setup(self, n_ts, sz, n_clusters):
        X = make_dataset(n_ts, sz, d=1)
        self.X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
        self.n_clusters = n_clusters

    def time_fit(self, n_ts, sz, n_clusters):
        KShape(
            n_clusters=self.n_clusters, n_init=1, max_iter=10,
            random_state=0, n_jobs=-1,
        ).fit(self.X)


if __name__ == "__main__":
    run_standalone(KMeansDTW, "TimeSeriesKMeans(metric='dtw')")
    run_standalone(KMeansSoftDTW, "TimeSeriesKMeans(metric='softdtw')")
    run_standalone(KShapeFit, "KShape")

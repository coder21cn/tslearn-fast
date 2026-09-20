"""Benchmarks for the parallel GAK fast path (used by TimeSeriesSVC and
the kernel KMeans assignment pass)."""
from tslearn.metrics import cdist_gak, sigma_gak

from ._common import make_dataset, run_standalone


class CdistGAK:
    params = (
        [60, 200],   # n_ts
        [64, 200],   # sz
        [1, 5],      # d
    )
    param_names = ["n_ts", "sz", "d"]

    def setup(self, n_ts, sz, d):
        self.X = make_dataset(n_ts, sz, d)
        # Estimate sigma once at setup so the benchmark times only the cdist.
        self.sigma = sigma_gak(self.X)

    def time_self(self, n_ts, sz, d):
        cdist_gak(self.X, None, sigma=self.sigma)


if __name__ == "__main__":
    run_standalone(CdistGAK, "cdist_gak")

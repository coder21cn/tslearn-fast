"""Benchmarks for Soft-DTW: pairwise cdist and the L-BFGS barycenter
optimisation that runs the fused parallel objective+gradient kernel.
"""
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import cdist_soft_dtw

from ._common import make_dataset, run_standalone


class CdistSoftDTW:
    params = (
        [60, 200],   # n_ts
        [64, 200],   # sz
        [1, 5],      # d
    )
    param_names = ["n_ts", "sz", "d"]

    def setup(self, n_ts, sz, d):
        self.X = make_dataset(n_ts, sz, d)

    def time_self(self, n_ts, sz, d):
        cdist_soft_dtw(self.X, None, gamma=1.0)


class SoftDTWBarycenter:
    params = (
        [30, 80],    # n_ts
        [64, 200],   # sz
        [1, 5],      # d
    )
    param_names = ["n_ts", "sz", "d"]

    def setup(self, n_ts, sz, d):
        self.X = make_dataset(n_ts, sz, d)

    def time_barycenter(self, n_ts, sz, d):
        softdtw_barycenter(self.X, gamma=1.0, max_iter=10)


if __name__ == "__main__":
    run_standalone(CdistSoftDTW, "cdist_soft_dtw")
    run_standalone(SoftDTWBarycenter, "softdtw_barycenter")

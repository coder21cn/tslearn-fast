"""Benchmarks for the parallel discrete Frechet cdist kernel."""
from tslearn.metrics import cdist_frechet

from ._common import make_dataset, run_standalone


class CdistFrechet:
    params = (
        [60, 200],   # n_ts
        [64, 200],   # sz
        [1, 5],      # d
    )
    param_names = ["n_ts", "sz", "d"]

    def setup(self, n_ts, sz, d):
        self.X = make_dataset(n_ts, sz, d)

    def time_unconstrained_self(self, n_ts, sz, d):
        cdist_frechet(self.X, None)

    def time_sakoe_chiba_self(self, n_ts, sz, d):
        cdist_frechet(
            self.X, None,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=max(2, sz // 20),
        )


if __name__ == "__main__":
    run_standalone(CdistFrechet, "cdist_frechet")

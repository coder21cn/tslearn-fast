"""Benchmarks for cdist_dtw across constraint shapes and dimensions.

Covers the three constraint types the fast path supports (unconstrained,
Sakoe-Chiba, Itakura) and walks the (n_ts, sz, d) shape grid that matters
in practice — short series with the per-pair wrapper used to dominate;
long series with band iteration are now bound by DP cost.
"""
from tslearn.metrics import cdist_dtw

from ._common import make_dataset, run_standalone


class CdistDTW:
    params = (
        [60, 200],         # n_ts
        [64, 200],         # sz
        [1, 5],            # d (1 = univariate, 5 = OHLCV-like)
    )
    param_names = ["n_ts", "sz", "d"]

    def setup(self, n_ts, sz, d):
        self.X = make_dataset(n_ts, sz, d)

    def time_unconstrained_self(self, n_ts, sz, d):
        cdist_dtw(self.X, None)

    def time_sakoe_chiba_self(self, n_ts, sz, d):
        cdist_dtw(
            self.X, None,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=max(2, sz // 20),
        )

    def time_itakura_self(self, n_ts, sz, d):
        cdist_dtw(
            self.X, None,
            global_constraint="itakura", itakura_max_slope=2.0,
        )


if __name__ == "__main__":
    run_standalone(CdistDTW, "cdist_dtw")

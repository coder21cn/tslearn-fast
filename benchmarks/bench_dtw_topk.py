"""Benchmarks for the LB_Keogh top-k DTW fast path.

This is the kNN/KMeans-assignment hot path. The full cdist_dtw reference
in the same shape range is the right baseline — the value of LB-prune is
how much it cuts the candidate set.
"""
from tslearn.metrics import cdist_dtw
from tslearn.metrics._dtw_lb import cdist_dtw_topk_fast

from ._common import make_dataset, run_standalone


class CdistDTWTopK:
    # Univariate UCR-style: the canonical workload the LB-prune fast path
    # was designed for. Multivariate is exercised by the parity tests; a
    # third grid axis here would balloon the cell count without adding
    # signal, since the kernel uses the same code path.
    params = (
        [50, 200],     # n_query
        [200, 500],    # n_candidates
        [100, 200],    # sz
    )
    param_names = ["n_query", "n_candidates", "sz"]

    def setup(self, n_query, n_candidates, sz):
        self.X = make_dataset(n_query, sz, d=1, seed=1)
        self.Y = make_dataset(n_candidates, sz, d=1, seed=2)
        self.radius = max(2, sz // 20)

    def time_top1_lb(self, n_query, n_candidates, sz):
        cdist_dtw_topk_fast(
            self.X, self.Y, k=1, radius=self.radius,
        )

    def time_top5_lb(self, n_query, n_candidates, sz):
        cdist_dtw_topk_fast(
            self.X, self.Y, k=5, radius=self.radius,
        )

    def time_full_cdist(self, n_query, n_candidates, sz):
        # Reference: the path the fast top-k replaces in kNN.
        cdist_dtw(
            self.X, self.Y,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=self.radius,
        )


if __name__ == "__main__":
    run_standalone(CdistDTWTopK, "cdist_dtw_topk_fast")

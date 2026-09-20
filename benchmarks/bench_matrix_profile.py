"""Benchmarks for the STOMP matrix-profile kernel.

The numpy backend used to extract every length-m subsequence and run an
O(n^2 * m) pairwise scan; the STOMP kernel does diagonal updates in
O(n^2). MatrixProfile remains univariate for main-parity compatibility.
"""
from tslearn.matrix_profile import MatrixProfile

from ._common import make_dataset, run_standalone


class MatrixProfileSTOMP:
    params = (
        [2500, 5000],   # series length
        [50, 200],      # subsequence length m
        [1],            # d (MatrixProfile is univariate)
    )
    param_names = ["L", "m", "d"]

    def setup(self, L, m, d):
        self.X = make_dataset(1, L, d)
        self.m = m

    def time_scaled(self, L, m, d):
        MatrixProfile(subsequence_length=self.m, scale=True).fit_transform(
            self.X
        )

    def time_unscaled(self, L, m, d):
        MatrixProfile(subsequence_length=self.m, scale=False).fit_transform(
            self.X
        )


if __name__ == "__main__":
    run_standalone(MatrixProfileSTOMP, "MatrixProfile (STOMP)")

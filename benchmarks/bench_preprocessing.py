"""Benchmarks for preprocessing transforms whose vectorized fast paths
were added during the perf sweep (PAA equal-length, TimeSeriesResampler
equal-length, size-1 nanmean)."""
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.preprocessing import TimeSeriesResampler

from ._common import make_dataset, run_standalone


class PAATransform:
    params = (
        [200, 1000],   # n_ts
        [200, 1000],   # sz
        [1, 5],        # d
        [10, 50],      # n_segments
    )
    param_names = ["n_ts", "sz", "d", "n_segments"]

    def setup(self, n_ts, sz, d, n_segments):
        self.X = make_dataset(n_ts, sz, d)
        self.paa = PiecewiseAggregateApproximation(n_segments=n_segments)

    def time_fit_transform(self, n_ts, sz, d, n_segments):
        self.paa.fit_transform(self.X)


class Resampler:
    params = (
        [200, 1000],   # n_ts
        [200, 1000],   # sz
        [1, 5],        # d
        [50, 100],     # target sz
    )
    param_names = ["n_ts", "sz", "d", "target"]

    def setup(self, n_ts, sz, d, target):
        self.X = make_dataset(n_ts, sz, d)
        self.target = target

    def time_transform(self, n_ts, sz, d, target):
        TimeSeriesResampler(sz=self.target).fit_transform(self.X)


if __name__ == "__main__":
    run_standalone(PAATransform, "PiecewiseAggregateApproximation")
    run_standalone(Resampler, "TimeSeriesResampler")

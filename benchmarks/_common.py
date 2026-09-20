"""Shared helpers for the asv benchmark suite.

Every benchmark module in this directory defines asv-style classes
(``setup`` + ``time_*`` methods) AND a ``__main__`` that prints a small
comparison table when run directly. The ``run_standalone`` helper here
is the wiring for that direct-invocation path so each benchmark stays
focused on the workload it measures.
"""
import time

import numpy


def make_dataset(n, sz, d=1, seed=0):
    """Synthetic random-walk-ish dataset of shape ``(n, sz, d)``."""
    rng = numpy.random.RandomState(seed)
    return rng.randn(n, sz, d).astype(numpy.float64)


def best_of(fn, repeat=3):
    """Best-of-``repeat`` wall-clock timing in seconds. Warms up once
    before sampling to amortize numba JIT and BLAS setup."""
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_standalone(benchmark_cls, label_prefix=""):
    """Iterate the cartesian product of ``benchmark_cls.params`` and print
    a one-line summary per ``time_*`` method. Lets a benchmark module
    work as a quick-check script without needing asv installed."""
    from itertools import product

    inst = benchmark_cls()
    params = getattr(benchmark_cls, "params", [[None]])
    names = getattr(benchmark_cls, "param_names", [""])
    time_methods = [
        m for m in dir(inst)
        if m.startswith("time_") and callable(getattr(inst, m))
    ]
    print(f"=== {label_prefix or benchmark_cls.__name__} ===")
    for combo in product(*params):
        try:
            inst.setup(*combo)
        except NotImplementedError:
            continue
        param_str = ", ".join(
            f"{n}={v}" for n, v in zip(names, combo)
        ) if names != [""] else ""
        for m in time_methods:
            method = getattr(inst, m)
            t = best_of(lambda: method(*combo))
            print(f"  {m:<32s} [{param_str:<40s}]  {t * 1000:8.2f} ms")

# tslearn Best Practices

A practical guide to when tslearn is the right tool, when it isn't, and how to scale it.

## When tslearn wins

Not in the "can't do" sense — general ML/DL can in principle do anything tslearn does. But tslearn has clear niches where it is faster, simpler, or more correct out of the box.

- **Elastic distance metrics out of the box.** DTW, Soft-DTW, GAK, LB_Keogh, and DTW Barycenter Averaging (DBA) are first-class and tuned (numba / joblib fast paths). General ML/DL libraries don't ship these; reimplementing correct, fast DTW with lower bounds is non-trivial.
- **Shape-based clustering.** k-Shape and DTW k-means with DBA centroids are the canonical use case. Deep clustering can match accuracy but needs orders of magnitude more data and compute. On a few hundred series, tslearn returns a sensible answer in seconds with no GPU.
- **Shapelet learning (Grabocka-style) with a scikit-learn API.** Possible to build in PyTorch, but tslearn gives you a one-liner.
- **Small-data regimes.** With 50–500 labeled series, kNN-DTW is often competitive with or beats deep models — no train loop, no GPU, no hyperparameter sweep. The fastest path to a working baseline.
- **Variable-length series.** DTW / GAK handle unequal lengths natively. Most DL pipelines force you to pad or resample first.
- **Interpretability + sklearn ergonomics.** Pipelines, `GridSearchCV`, `fit/predict`, deterministic results. Useful when you need to defend a model, not just deploy one.
- **No training infrastructure.** kNN-DTW, KShape, SAX / PAA representations — zero training for the lazy learner, CPU-only, reproducible.

## When tslearn does *not* win

- Large datasets (see thresholds below).
- Forecasting — use `statsforecast` / Nixtla / Darts.
- Long sequences where DTW's O(n·m) is painful.
- Anywhere a transformer / TCN with enough data will dominate on accuracy.

**Rule of thumb:** under ~1k series, or when you need elastic similarity / shape-based clusters, reach for tslearn first. Above that and with labels, DL usually wins on accuracy if not on effort.

## Scaling thresholds

Order-of-magnitude guidance, not hard cutoffs.

### Dataset size (number of series, N)

| Method | Comfortable | Painful past |
|---|---|---|
| kNN-DTW / pairwise DTW | a few thousand | ~10k (O(N²) pairwise, O(n·m) per pair) |
| KShape, TimeSeriesKMeans + DBA | ~10k–50k | DBA is the bottleneck — iterative DTW alignments per cluster per iteration |
| Shapelet learning | ~1k–5k | candidate-shapelet space blows up |

Above roughly **10⁵ series**, switch to sktime's index-based methods, `tsfresh` features + a GBM, or a deep model on GPU.

10k × 10k of length-256 series ≈ 6×10¹¹ cell ops even with numba — plan accordingly.

### Sequence length (n)

| Length | Notes |
|---|---|
| < 256 | trivial |
| 256 – 1,024 | tslearn's sweet spot |
| 1k – 5k | use Sakoe-Chiba band (`sakoe_chiba_radius`) or Itakura constraint — a single unconstrained DTW at length 4k is ~16M cell ops |
| 5k – 10k | unconstrained is painful; with a tight band (radius ≈ 5–10% of length) still usable |
| > 50k | downsample, or use PAA / SAX first, or move to DL (TCN, transformer) — linear-ish in length |

LB_Keogh-pruned kNN extends the practical range significantly for nearest-neighbor search specifically.

### Multivariate series (d channels)

- **Supported everywhere.** Shape is `(n_ts, sz, d)`. DTW, Soft-DTW, GAK, kNN, KShape, TimeSeriesKMeans, and shapelets all accept `d > 1`.
- **It's *dependent* DTW.** One shared warping path across all channels, computed on the multivariate Euclidean distance between frames.
  - Right model when channels are synchronized (e.g. x/y/z accelerometer).
  - Wrong model when channels warp independently (e.g. unrelated sensors). For that, loop per channel and sum (independent-DTW) — not shipped as a one-liner.
- **Cost grows linearly in `d`.** Each cell is a d-dim distance. 10 channels ≈ 10× the runtime of univariate. High-d + long sequences is where DL pulls ahead fastest.
- **Preprocessing matters more.** Use `TimeSeriesScalerMeanVariance` per-channel; otherwise a single high-variance channel dominates the DTW cost.

### Cost heuristic

If `N² · n · m · d` exceeds **~10¹⁰ – 10¹¹**, plan for bands, LB_Keogh pruning, downsampling, or a different tool.

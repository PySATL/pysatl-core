# Bootstrap

[Russian version](bootstrap.ru.md).

## Overview

Bootstrap estimates the variability of a statistic or parameter estimate through
repeated samples. The same operation with the same settings is applied to each
new sample. The distribution of the resulting values can be used, for example,
to estimate the standard error of the sample mean or a parameter estimate.

Two modes are considered for a univariate i.i.d. sample of size `n`:

- **Nonparametric:** for each replicate, draw `n` original observations with
  replacement. Repeated values retain their weight: for `[1, 1, 4]`, the
  probabilities of values `1` and `4` are `2/3` and `1/3`.
- **Parametric:** fit a model to the original sample and generate `n` observations
  from it for each replicate. The generating model remains fixed across
  replicates; when studying an estimator, fit it again on each new sample.

## Requirements and architectural support

| Bootstrap requirement | Existing support and points to clarify |
|---|---|
| Preparing the original and replicate samples | The RFC provides `Sample.prepare` and a protected observation snapshot. Each replicate can be prepared independently. |
| Repeated statistic computation | `Statistic.compute(sample)` applies one configuration to different samples. `SampleSession` provides dependencies, caching and diagnostics; when sessions are used, each replicate needs its own session. |
| Repeated parameter estimation | The existing estimation workflow with a family, estimator and fit result supports repeated fitting. Integration must align data transfer from `Sample` and extraction of numeric estimates, status and the fitted distribution. |
| Nonparametric sampling | Drawing indices of the original observations with replacement is sufficient. This is the exact empirical measure; constructing a separate ECDF is optional, while KDE defines a different generating distribution. |
| Parametric sampling and reproducibility | `Distribution.sample` and sampling strategies are available. However, the default strategy creates its own RNG, and UNU.RAN uses a shared URNG. A way to pass a controlled generator and verify support for the required models is needed. |
| Replicate loop and aggregate result | Bootstrap must implement the replicate count, operation calls, collection of values and diagnostics, summaries and failure accounting. These responsibilities belong to neither the statistic nor the estimator. |

**Conclusion:** the architectural roles allow Bootstrap to be integrated without
changing the core statistics and estimation workflows. A complete implementation
still needs replicate management, aligned module inputs/results and sampling
with a controlled RNG.

A replicate can be degenerate: for example, `[1, 1, 1]` does not allow a
nondegenerate Normal fit with both parameters estimated. Failures and their
causes must be recorded, continuation rules defined, and summaries based only
on successful replicates explicitly identified as such.

## Implementation choice

The algorithms, public API, result format and failure-handling rules will be
selected in the **[Bootstrap implementation task](https://github.com/PySATL/pysatl-core/issues/145)**.

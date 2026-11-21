# Empirical distributions: E01–E02

[General conditions](README.md#general-conditions-for-reading-the-cards).
E01/E02 construct a distribution with available characteristics and fitting
metadata.

Construction is deterministic; sampling requires an explicit RNG.
The data snapshot and caches are tied to the configuration. Changing the method
must replace the characteristics and support consistently, preserving the previous
state if the change fails. Fitting metadata records the sample size, configuration,
distribution type/support, available characteristics and reasons for numerical failures.
Weighted and censored data and streaming updates are outside the current scope.

## E01. ECDF and empirical PMF

Constructs a discrete measure: `F_n(z)=sum(1[x_i<=z])/n`,
`p_n(z)=sum(1[x_i=z])/n`. The ECDF gives the proportion of observations no greater
than z; the PMF gives the proportion equal to z.

- **Input and dependencies:** a nonempty sample retaining multiplicities and,
  optionally, an explicit support to validate; grouping/sorting and Distribution strategies.
- **Computation:** one possible implementation is a table of unique values,
  counts and cumulative probabilities. This approach is used by `scipy.stats.ecdf`
  for uncensored samples
  ([source code of `_ecdf_uncensored`](https://github.com/scipy/scipy/blob/main/scipy/stats/_survival.py)).
  Constant samples and noninteger observations are allowed. Merging nearby floating-point
  values changes the measure and must not happen silently. The CDF is right-continuous,
  and the total probability mass is 1.
- **Reuse:** in the table-based implementation, the table is reused for repeated queries;
  sampling preserves empirical frequencies. Sorting-based preparation costs `O(n log n)`,
  the table requires `O(u)` storage, and the original snapshot requires another `O(n)`.

**Example.** For `[1,1,3.5]`, the masses are `2/3` and `1/3`:
`F_n(2)=2/3`, `p_n(2)=0`. The support need not be an integer lattice.

**Source:** [SciPy ecdf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html).

## E02. Gaussian KDE

Constructs a smooth mixture of Gaussian kernels centered at the observations:
`f_h(z)=mean(phi((z-x_i)/h))/h`, `F_h(z)=mean(Phi((z-x_i)/h))`.
The support is the entire real line.

- **Input and dependencies:** a sample, a bandwidth rule/factor or an absolute
  bandwidth h; Gaussian PDF/CDF and Distribution.
- **Computation and settings:** store the selection rule and the actual `h>0`
  separately; a standard-deviation multiplier is not an absolute bandwidth.
  Grouping repeated observations must preserve their masses and the original
  bandwidth estimate. Direct PDF/CDF evaluation at m points costs `O(nm)`;
  processing in blocks limits temporary memory.
- **Specific requirements:** the current SciPy path requires `n>=2` and positive spread.
  A constant sample with an externally specified h still defines a mathematical
  mixture; the implementation restriction must not be hidden by adding noise.
  Bounded support requires boundary correction. Integration/PPF need separate
  accuracy checks. The KDE mean equals the sample mean, and its variance is
  `var(x,ddof=0)+h²`; sampling adds Gaussian noise and differs from ECDF resampling.

**Example.** For `[0,2]` and `h=1`, an equally weighted mixture of two Normal
distributions with variance 1 has mean 1 and variance 2; the empirical measure
has variance 1.

**Source:** [SciPy gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

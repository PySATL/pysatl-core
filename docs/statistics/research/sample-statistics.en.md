# Sample statistics: research S0–S13

[Russian version](sample-statistics.ru.md) · [RFC](../rfc/rfc.md).

The cards describe the mathematical meaning of the statistics, significant
parameters, applicability conditions, and requirements that affect the architecture.
General rules are collected in the [conclusions](#conclusions); the RFC defines
the API details. The selected implementations are preliminary.

Notation: `n` is the number of observations, `x_i` is an observation, and `r` is
the moment order. `x_(i)` is the i-th value after [sorting with S0](#s0-array-sorting);
mathematical ranks start at 1. `C(a,b)` is the number of ways to choose `b`
elements from `a`.

## S0. Array sorting

A helper function for computing statistics: returns the array of observations
sorted in ascending order. Duplicates are preserved; the original array is unchanged.

**Example:** `[8, 2, 1, 2]` → `[1, 2, 2, 8]`.

**Implementation notes:** sorting is a separate helper function so that, when
caching is enabled, the session can cache the sorted array and reuse it in
multiple calculations on the same sample without sorting again.

**Selected implementation:** `numpy.sort`.

## S01. Sample mean

The mean describes the location of the data: it is the sum of all observations
divided by their count. Values with large absolute magnitudes can noticeably
shift the result.

**Formula:** `mean = sum(x_i) / n`.

**Example:** `[1, 2, 4, 8]` → `15/4 = 3.75`. Replacing `8` with `80`
increases the mean to `21.75`.

**Parameters and conventions:** —

**Numerical consideration:** the sum may overflow before division by `n`.
Scaling can help avoid this: divide the values by a suitable positive number
`x`, compute the mean, then multiply it by `x`. The original sample remains
unchanged; this does not reduce memory use.

**Selected implementation:** the standard NumPy mean.

**Comparison reference:** [NumPy mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html).

## S02. Sample variance

Variance measures dispersion around the mean by averaging squared deviations.
It is expressed in squared units of the original measurements; its square root,
the standard deviation, has the original units.

**Formula:** `variance = sum((x_i - mean)^2) / (n - ddof)`.

**Parameters and conventions:** `ddof` is the number subtracted from the
observation count in the divisor. With `ddof=0`, divide by `n`; with `ddof=1`,
divide by `n−1`. The first option describes the dispersion of the sample itself;
the second gives an unbiased estimate of the distribution's variance for
independent, identically distributed observations with finite variance.
The initial version supports integer `ddof >= 0`.

**Example:** for `[1, 2, 3]`, the mean is `2` and the sum of squared deviations
is `2`. The result is `2/3` with `ddof=0` and `1` with `ddof=1`.

**Conditions:** `n > ddof`; otherwise, the divisor is invalid. A constant sample
with a valid divisor has zero variance.

**Implementation notes:** `ddof` is a parameter of the specific statistic:
it changes the result and the minimum required sample size. The calculation
requires the mean [S01](#s01-sample-mean).

**Selected implementation:** compute the mean, then the squared deviations,
then divide by `n−ddof`. The formula `mean(x_i^2) - mean(x_i)^2` is not used:
it can lose accuracy for large values with small dispersion.

**Comparison reference:** [NumPy var](https://numpy.org/doc/stable/reference/generated/numpy.var.html).

## S03. Raw moments

The raw moment of order `r` is the mean of the values raised to the power `r`.
The first moment equals the mean; the second is the mean square, rather than
the variance. The order specifies the power applied to the values; higher
orders give greater emphasis to observations with large absolute magnitudes.

**Formula:** `m_r = mean(x_i^r)`.

**Example:** for `[1, 2, 3]`, we obtain `m_1=2`, `m_2=14/3`, `m_3=12`.

**Parameters and conventions:** one or more integer orders `r >= 0`.
The zeroth moment is defined as `1`, including for samples containing zeros.
Negative and fractional orders are outside the initial version's scope.

**Implementation notes:** requests for multiple orders preserve their sequence
and duplicates. These moments are also needed by the method of moments and
the generalized method of moments in the estimation module.

**Selected implementation:** for each order separately, compute powers of the
observations and their mean with NumPy. High powers may overflow.

**Comparison reference:** [SciPy moment with center=0](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html).

## S04. Central moments

The central moment of order `r` is the mean of the deviations from the sample
mean raised to that power. Adding the same constant to every observation
leaves it unchanged. The second central moment equals the variance with
divisor `n`. The third and fourth are used to describe asymmetry and
distribution shape.

**Formula:** `c_r = mean((x_i - mean(x))^r)`.

**Example:** for `[1, 2, 3]`, we obtain `c_2=2/3`, `c_3=0`.
The values are the same for `[101, 102, 103]`.

**Parameters and conventions:** one or more integer orders `r >= 0`;
the center is strictly the sample mean. Mathematically, `c_0=1`, `c_1=0`.
No `ddof` correction is applied to `c_2`.

**Implementation notes:** uses the mean [S01](#s01-sample-mean).

**Selected implementation:** compute the mean and deviations, then average
their powers. Recovering central moments from raw moments can lose accuracy;
high powers may overflow.

**Comparison reference:** [SciPy moment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html).

## S05. Sample quantiles

The quantile at level `p` describes a boundary for the lower proportion `p`
of the data. For a finite sample, the level alone is insufficient: a rule
must specify whether to select an observation or a value between adjacent
observations.

**Linear interpolation formula:** the position in a zero-indexed array is
`h=(n−1)p`. With `j=floor(h)` and `d=h−j`, the result is
`(1−d) × sorted[j] + d × sorted[j+1]`; if `d=0`, use `sorted[j]`.
`floor` means rounding down, and `sorted` is the sorted sample.

**Example:** for `[1, 2, 4, 8]` at `p=0.5`, linear interpolation gives `3`,
whereas the inverse empirical cumulative distribution function (`inverted_cdf`)
gives `2`. At `p=0.75`, linear interpolation gives `5`.

**Parameters and conventions:** one or more levels `0 <= p <= 1` and a quantile
definition method. For the methods described here, levels `0` and `1` return
the minimum and maximum. Levels may be repeated and given in any order.

**Implementation notes:** the method is part of the statistic's definition
and must not change in response to a numerical error. Result values must
remain aligned with their requested levels.

**Selected implementation:** [S0](#s0-array-sorting) followed by the formula
for the selected convention.

**Comparison reference:** [NumPy quantile and Hyndman–Fan conventions](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).

## S06. Order statistics

An order statistic is the observation at a specified position after sorting.
The first is the minimum; the last is the maximum. Unlike an interpolated
quantile, the result is always an original observation.

**Example:** sorting `[8, 1, 4, 2]` gives `[1, 2, 4, 8]`.
API index `2` selects `4`. For `[2, 2, 5]`, indices `0` and `1` both return `2`.

**Parameters and conventions:** one or more indices. The public API provisionally
uses zero-based indexing: `0 <= index < n`. Mathematical ranks in formulas
start at 1.

**Implementation notes:** duplicate observations occupy separate positions
and must not be removed before calculation. The order and duplicates of
requested indices are preserved.

**Selected implementation:** select elements from the result of [S0](#s0-array-sorting).

**Comparison reference:** [NumPy partition](https://numpy.org/doc/stable/reference/generated/numpy.partition.html), comparing the selected values.

## S07. Median

The median is the middle of the ordered data: the central observation for odd
`n`, or the mean of the two central observations for even counts. Values far from
the center affect it less than they affect the mean.

**Example:** for `[1, 2, 3, 4, 100]`, the median is `3` and the mean is `22`.
For `[1, 2, 4, 8]`, the median is `(2+4)/2 = 3`.

**Parameters and conventions:** —

**Important distinction:** the rule for even `n` is fixed in the definition
above. It differs from the quantile at `p=0.5` under the inverse empirical
cumulative distribution function, which gives `2` for `[1, 2, 4, 8]`.

**Selected implementation:** select the central element or average the two
central elements from the result of [S0](#s0-array-sorting).

**Comparison reference:** [NumPy median](https://numpy.org/doc/stable/reference/generated/numpy.median.html).

## S08. Trimmed mean

The trimmed mean is the mean after removing specified proportions of the
smallest and largest observations. Left and right trimming may differ.

**Formula:** for proportions `a` on the left and `b` on the right, remove
`l=floor(n×a)` and `u=floor(n×b)` observations, then compute the mean of the
remaining observations. `floor` means rounding down.

**Example:** trimming 20% from each side of `[1, 2, 3, 4, 100]` leaves
`[2, 3, 4]`, whose mean is `3`. For four observations, the same proportions
remove nothing: `floor(4×0.2)=0`.

**Parameters and conventions:** separate proportions `a,b >= 0`; the number
of observations removed is rounded down. At least one must remain: `l+u < n`.

**Implementation notes:** —

**Selected implementation:** [S0](#s0-array-sorting), select the retained range,
then compute its mean.

**Comparison reference:** [SciPy trim_mean](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html) for symmetric trimming.

## S09. Probability-weighted moments (PWM)

For a continuous distribution, the PWM of order `r` is defined as
`E[X × F(X)^r]`, where `F(X)` is the probability of obtaining a value no greater
than `X`. The probability is raised to a power, rather than the observation.
In the selected sample estimator, higher ranks receive greater weight for
`r>0`. At `r=0`, this reduces to the ordinary mean. PWM are used to compute
L-moments.

**Sample formula:**
`b_r = (1/n) sum_{i=r+1}^n [C(i−1,r)/C(n−1,r)] × x_(i)`.
Its interpretation is the mean maximum of all subsamples of size `r+1`, divided
by `r+1`. Subsamples are selected without replacement.

**Example:** for `[1, 2, 4, 8]` and `r=1`, the multipliers are `[0, 1/3, 2/3, 1]`.
Thus `b_1=(0×1 + (1/3)×2 + (2/3)×4 + 1×8)/4 = 17/6`.
The multipliers are not normalized to sum to one.

**Parameters and conventions:** integer orders `r >= 0`, each requiring
`n >= r+1`. The combinatorial estimator in the formula above is selected.

**Important distinction:** for a distribution with point masses, this formula
estimates `E[max(X_1,…,X_(r+1))]/(r+1)`, which may differ from the literal
`E[X × F(X)^r]`. For example, if `X` always equals `1`, the values are
`1/(r+1)` and `1`, respectively. Substituting the empirical cumulative
distribution function for `F` also defines a different estimator.

**Implementation notes:** the PWM convention must be explicit and consistent
with the L-moments [S11](#s11-l-moments). Different definitions must not be
presented as the same method.

**Selected implementation:** [S0](#s0-array-sorting), then compute multipliers
recursively from higher to lower ranks without constructing large binomial
coefficients.

**Comparison reference:** [Greenwood et al., 1979](https://doi.org/10.1029/WR015i005p01049),
[NIST L-moments](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/lmoment.htm).

## S10. General L-statistics

An L-statistic is a linear combination of ordered observations. A weight applies
to a position in the sorted sequence, rather than to a particular observation
before sorting.

**Formula:** `L = sum(w_i × x_(i))`.

**Example:** for `[1, 2, 4, 8]`, weights `[1/2, 0, 0, 1/2]` give the midpoint
between the minimum and maximum, `4.5`. Weights `[-1/2, 0, 0, 1/2]` give
half the range, `3.5`.

**Parameters and conventions:** weights or a rule for generating them for size
`n`. Weights must be finite and their count must equal `n`. They may be negative
and need not sum to `1`. Multiple sets of weights may be requested.

**Implementation notes:** rank weights are part of the statistic's definition;
they do not require probability normalization. Explicit weights are tied to
the sample size; a generation rule must determine them unambiguously for
a given `n`.

**Selected implementation:** the dot product of the weights and the result
of [S0](#s0-array-sorting) with NumPy, one set of weights at a time.

**Comparison reference:** [NumPy dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html):
`np.dot(weights, np.sort(sample))` with the same rank weights.

## S11. L-moments

L-moments are L-statistics with specific weights. The first equals the mean;
the second describes dispersion and equals half the mean absolute difference
over all pairs of observations with distinct indices. The third and fourth
are used to describe asymmetry and shape after normalization. Observations
are not raised to high powers, but extreme values may still affect the result.

**Formula via PWM from [S09](#s09-probability-weighted-moments-pwm):**
`l_r = sum_{j=0}^{r−1} (−1)^(r−1−j) × C(r−1,j) × C(r−1+j,j) × b_j`.
In particular, `l_1=b_0`, `l_2=2b_1−b_0`.

**Example:** for `[1, 2, 4, 8]`, we have `l_1=15/4`.
The sum of the six absolute pairwise differences is `23`, so
`l_2=(23/6)/2 = 23/12`. Unlike variance, `l_2` has the same units as
the original data.

**Parameters and conventions:** integer orders `r >= 1`, requiring `n >= r`.
Unbiased estimators through combinatorial PWM are selected. For a constant
sample, `l_1` equals that constant and the other L-moments are zero.
A finite `E|X|`, the expected absolute value, ensures the existence of all
corresponding population moments.

**Implementation notes:** uses PWM from [S09](#s09-probability-weighted-moments-pwm)
with the same convention; sorting [S0](#s0-array-sorting) takes place when
computing PWM. L-moments themselves and their dimensionless ratios
[S12](#s12-l-moment-ratios) are different results.

**Selected implementation:** transform PWM into L-moments using the formula above.

**Comparison reference:** [SciPy stats.lmoment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html)
(SciPy >= 1.15): `scipy.stats.lmoment(sample, order=orders, standardize=False)`.
The comparison uses L-moments themselves; with `standardize=True`, orders 3
and above are returned as ratios to the second L-moment.

## S12. L-moment ratios

L-moment ratios are dimensionless characteristics of relative dispersion
and shape. The following must be distinguished:

- `l_2/l_1` is the L-coefficient of variation: dispersion relative to the mean level.
- `l_3/l_2` is L-skewness.
- `l_4/l_2` is L-kurtosis, a tail-shape characteristic; it is not ordinary excess kurtosis.

The shape ratios `l_r/l_2` for `r>=3` are unchanged by adding a constant
or multiplying the data by a positive number. The ratio `l_2/l_1` depends on
the origin; when `l_1` is negative, its usual positive interpretation as
relative dispersion no longer holds.

**Example:** for `[1, 2, 4, 8]`, the values `l_1=15/4`, `l_2=23/12`, `l_3=3/4`
give `l_2/l_1=23/45`, `l_3/l_2=9/23`. For `[3, 3, 3]`, the denominator
`l_2=0`, so L-skewness is undefined.

**Parameters and conventions:** select `l_2/l_1` and/or orders `r>=3` for
ratios `l_r/l_2`. Sample-size requirements are those of the participating
L-moments [S11](#s11-l-moments); the denominator must be nonzero.

**Implementation notes:** uses L-moments [S11](#s11-l-moments). Applicability
depends on the specific ratio: a zero `l_1` rules out `l_2/l_1`, but not
necessarily `l_3/l_2`. If one component fails, the entire vector request is
rejected with its position reported; valid statistics requested separately
remain available.

**Selected implementation:** divide the required L-moments. A zero denominator
is an error; a near-zero denominator can amplify numerical error. No universal
absolute cutoff is introduced for a nonzero denominator, since it would depend
on the units of the data.

**Comparison reference:** [Lmo l_ratio](https://jorenham.github.io/Lmo/api/L-moments/):
`lmo.l_ratio(sample, 2, 1, trim=0)` for `l_2/l_1`,
`lmo.l_ratio(sample, r, 2, trim=0)` for `l_r/l_2`.

## S13. Trimmed L-moments (TL)

TL-moments reduce the influence of extreme ranks in the definition of L-moments.
Left trimming `s` and right trimming `t` are specified separately. This differs
from computing ordinary L-moments after removing extreme observations from
the original sample.

**Mathematical interpretation:** for order `r`, consider all subsamples of size
`M=r+s+t`, selected without replacement. Let `A_(a,M)` be the mean of the a-th
order statistics of these subsamples. Then
`l_r^(s,t) = (1/r) sum_{j=0}^{r−1} (−1)^j × C(r−1,j) × A_(r+s−j,M)`.
Without enumerating subsamples, this mean is expressed as
`A_(a,M) = sum_{i=1}^n C(i−1,a−1) × C(n−i,M−a) × x_(i) / C(n,M)`.
Binomial coefficients outside the valid bounds are taken as zero.

**Example:** the first TL-moment with `s=t=1` is the mean of the medians of
all triples. For `[0, 1, 2, 10, 100]`, the medians of the ten triples equal
`1` three times, `2` four times, and `10` three times. The result is
`(3×1+4×2+3×10)/10 = 4.1`. Simply removing `0` and `100` instead gives
a retained mean of `13/3 ≈ 4.333`, which is a different statistic.

**Parameters and conventions:** integer orders `r >= 1` and independent
nonnegative integer trimming amounts `s,t`. Requires `n >= r+s+t`.
The unbiased estimator in the formula above is selected. With `s=t=0`,
this gives ordinary L-moments; fractional trimming is outside the initial
version's scope.

**Implementation notes:** the order and both trimming amounts jointly determine
the minimum sample size. Trimming changes rank weights, so a TL-moment must
not be replaced with an L-moment of a truncated sample. Zero weights at extreme
ranks alone do not reduce `n_used`.

**Interpretation note:** some population TL-moments exist even when the ordinary
mean does not. Existence concerns the expected values of the required internal
order statistics, rather than the general condition of finite `E|X|`.

**Selected implementation:** [S0](#s0-array-sorting), then the direct formula
using binomial coefficients (`math.comb`), one order at a time.

**Comparison reference:** [Elamir–Seheult, 2003](https://durham-repository.worktribe.com/output/1560955),
[Hosking, 2007](https://doi.org/10.1016/j.jspi.2006.12.002).

## Conclusions

- **Common input:** a one-dimensional, nonempty sample of finite real numbers.
  `Sample.prepare` creates a protected `float64` snapshot, preserving order
  and duplicates. Requirements of individual statistics, such as `n > ddof`,
  are checked by the statistic itself.
- **Result shape:** the public result is `StatisticResult`; a number or vector
  is held in its `value` field. A vector preserves the order and duplicates
  of the request. Empty requests are rejected. If a component cannot be
  computed, the entire request fails with its position reported.
- **Significant parameters:** orders, levels, indices, divisor correction,
  trimming, weights, and mathematical conventions distinguish requests and
  form part of the cache key. They must remain unchanged throughout the
  lifetime of the statistic object.
- **Shared rank preparation:** quantiles, order statistics, the median, and
  L-/TL-methods use [S0](#s0-array-sorting). Its result can be shared across
  calculations through the RFC dependency mechanism; the original snapshot
  remains unchanged.
- **Initial version's scope:** observation weights and streaming are not
  supported. Rank weights [S10](#s10-general-l-statistics) and trimming
  [S08](#s08-trimmed-mean)/[S13](#s13-trimmed-l-moments-tl) belong to the
  definitions of the statistics and are not weights of the input Sample.
- **Sample and distribution:** these statistics do not require a probability
  distribution family or fitting its parameters. A finite sample value does
  not prove that the corresponding population moment exists. Independence
  and identical distribution of observations are not checked automatically.
- **Numerical errors:** finite input values do not guarantee a computable
  result with limited precision. The RFC defines the common error rules;
  observations must not be silently removed, mathematical conventions must
  not be substituted, and non-numeric or infinite values must not be reported
  as successful results.

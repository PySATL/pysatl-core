# Parametric methods: P01–P11

How the methods fit into the architecture described in [the course paper](../../vkr.pdf),
sections 5.1–5.6. The documented prototype implements MLE; the other entries are
proposed extensions. The general workflow `ParametricFamily.fit → Estimator[R] → FitResult`
is retained; no new `EstimationTarget` layer is introduced.

## Common integration

- Each method is a separate `Estimator` with an `estimate(family, sample)` operation.
  Settings belong to the estimator; adding a method does not change `family.fit`.
- Reuse problem-preparation facilities for free parameters/views, packing into
  a numeric vector, initialization, bounds and validity checks. Sample requirements
  are method-specific: the prototype's `n>=d` check must not be applied mechanically
  to every estimator.
- The result extends `FitResult` with method-specific data: the criterion,
  residuals or equation residuals. `LogLikelihood` and `MLEResult` are specific to MLE;
  log-PDF and AIC/BIC are not common requirements for all estimators.
- The existing `Optimizer.minimize` accepts a scalar function and supports new
  criteria. Root finding requires a separate numerical mechanism. Check the original
  criterion/residuals, parameters and support; retain settings and stopping reasons.
  Local convergence does not prove uniqueness.
- The prototype's general limitations remain: incomplete support for converting
  results between parameterizations and a heuristic for determining whether support
  depends on parameters. These are discussed in sections 5.4–5.6 and 7.3 of the paper.

## P01. Maximum likelihood estimation (MLE)

An `Estimator` maximizes the sum of log-likelihood contributions. It already uses
`FitProblem`, `LogLikelihood`, `EstimationFormulaRegistry`, `Optimizer` and `MLEResult`.

**Additions:** Gamma can use a specialized scalar-equation solution instead of
general minimization. This needs root finding and provenance identifying that path;
the estimator invocation workflow remains unchanged.

[Gamma MLE description](https://tminka.github.io/papers/minka-gamma.pdf).

## P02. Arbitrary pointwise contrast

An `Estimator` constructs `mean(rho(x_i,theta))` and passes the function to the existing
`Optimizer.minimize`. The result records the loss, its settings and the criterion value.

**Additions:** a user-defined loss contract and declarations of the model operations
it needs. `LogLikelihood` is not used; no new general search mechanism is required.

## P03. Least squares

An `Estimator` computes residuals `r(theta)` and minimizes `sum(w*r²)` with the existing
optimizer; the result also records the residuals, weights and sum of squares.

**Additions:** a residual function and weight validation. A specialized least-squares
solver can be added separately; it is optional for the basic path.
Nonzero residuals at the minimum are allowed.

[SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## P04. Minimum CDF distance, including CvM

An `Estimator` prepares sorting/the ECDF, obtains the distribution's CDF for each
parameter candidate and minimizes the selected distance with the existing optimizer.

**Additions:** a criterion evaluator with an explicit measure and normalization.
Continuous CvM has a finite-sum formula; a discrete model requires a different formula.
Integration is needed only for criteria without an available direct computation.

[SciPy CvM](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html).

## P05. PMF and frequency distances

An `Estimator` prepares groups/frequencies once and minimizes their discrepancy
from model probabilities through `Optimizer.minimize`; PMF or CDF is needed for the groups.

**Additions:** grouping, the remaining probability mass outside the listed categories
and rules for zero probabilities. The common estimation interface need not change.

## P06. Generalized method of moments (GMM)

An `Estimator` computes averaged conditions `g(theta)` and minimizes
`g(theta)^T W g(theta)` with the existing optimizer.

**Additions:** condition functions, validation of the weight matrix W and diagnostics
for the conditions/weights. The basic path uses a fixed W; estimating it and performing
a two-step update is a separate procedure. With singular W, a zero criterion does not
imply zero residuals for all conditions.

[Hansen, 1982](https://ww.e-m-h.org/Hans82.pdf).

## P07. Estimating equations

An `Estimator` constructs the system `Psi(theta)=0`, reuses parameter packing
and returns an extension of `FitResult` with residuals and root-finding diagnostics.

**Additions:** a separate scalar/vector root-finding mechanism, its tolerances,
parameter-domain handling and rules for missing or multiple roots.
The existing `Optimizer.minimize` does not replace this contract: a minimum of the
squared residual may not be a root. Success requires small residuals of the equations.

[SciPy root](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html),
[root_scalar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html).

## P08. Robust M-estimation of location

An `Estimator` minimizes `sum(rho((x_i-mu)/s))` with the existing optimizer.
The influence-equation variant uses the mechanism from P07.

**Additions:** a configurable loss/influence function and a rule for positive scale s,
for example using the median and MAD. Zero MAD requires an explicit policy;
joint estimation of location and scale is a separate problem.

[Huber, 1964](https://doi.org/10.1214/aoms/1177703732).

## Common integration of moment methods P09–P11

An `Estimator` computes sample characteristics once and matches them to model
characteristics using a formula, root finding from P07 or discrepancy minimization
from P06. Model-characteristic evaluators are needed when the family does not provide
them directly; integration requires checks of existence and accuracy.
The result records the selected characteristics and matching residuals.

## P09. Ordinary method of moments (MoM)

An `Estimator` matches sample and theoretical moments using the common workflow above.
Direct estimates can be registered in `EstimationFormulaRegistry` for this method;
a moment-based MLE starting point is not itself a standalone MoM estimate.

**Additions:** sample moments and access to theoretical model moments.
The support check needs an explicit policy: a moment-based Uniform estimate may not
cover every observation; silently expanding the resulting interval is not allowed.

[SciPy fit: MM](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html).

## P10. Estimation by L-moments

An `Estimator` uses the same matching workflow for L-moments and their ratios.

**Additions:** sample L-moments and theoretical L-characteristics, provided by direct
formulas or computed through PPF and integration. Missing computations must be added;
having a PPF does not itself provide them. The model's ordinary variance need not exist.

[Hosking, 1990](https://doi.org/10.1111/j.2517-6161.1990.tb01775.x).

## P11. Estimation by trimmed L-moments

An `Estimator` matches TL-moments using the same workflow, with identical left and right
trimming `(s,t)` for the sample and the model.

**Additions:** sample and theoretical TL-moments for the selected trimming,
their existence conditions and numerical accuracy. This extends characteristic
evaluators; no new estimator interface is required. With `(s,t)=(0,0)`,
the result is consistent with ordinary L-moments.

[Hosking, 2007](https://doi.org/10.1016/j.jspi.2006.12.002),
[Lmo — a possible TL reference](https://jorenham.github.io/Lmo/api/L-moments/).

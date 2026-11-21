# Estimation research

Research on empirical distributions and integration of parametric estimators
for [task #137](https://github.com/PySATL/pysatl-core/issues/137).
The [existing estimation architecture is retained](../rfc/README.md);
`EstimationTarget` is not introduced. The parametric methods are mapped to
[the course paper](../../vkr.pdf), sections 5.1–5.6.

## Documents

| Document | Russian | English | Contents |
|---|---|---|---|
| Empirical distributions | [Russian](empirical-distributions.ru.md) | [English](empirical-distributions.en.md) | E01–E02: ECDF/PMF and Gaussian KDE. |
| Parametric estimators | [Russian](parametric-estimators.ru.md) | [English](parametric-estimators.en.md) | P01–P11: each estimator's work, reusable components and required additions. |

## General conditions for reading the cards

- Input is a nonempty one-dimensional sample of finite numeric observations.
  Independence and identical distribution are user assumptions.
- Notation: `n` is the sample size, `d` the number of free scalar parameters,
  `u` the number of unique observations and `m` the number of query points.
  A vector of characteristics or parameters does not imply multivariate data.
- Sample-size requirements depend on the method and model; `n>=d` does not
  guarantee that an estimate exists. Formula applicability depends on the
  selected parameterization and fixed parameters.
- Invalid inputs, settings and missing model capabilities are checked before
  numerical work; candidate-dependent restrictions are checked during evaluation.
  Finite inputs do not guarantee finite intermediate results. Failed attempts,
  changes to the criterion and discarded observations must not be hidden.
- Configuration and mathematical conventions are explicit. Numerical tolerances
  are implementation decisions. Deterministic calculations need no RNG;
  random search, restarts and sampling require an explicit generator.

[Estimation](../README.md).

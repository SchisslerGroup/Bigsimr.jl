# Bigsimr

| **Documentation**                       | **Build and Coverage**                    | **Package Details**                    |
|:---------------------------------------:|:-----------------------------------------:|:--------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] | [![CI][ci-img]][ci-url]                   | [![Licence][license-img]][license-url] |
| [![][docs-latest-img]][docs-latest-url] | [![Coverage][codecov-img]][codecov-url]   | ![Release][release-img]                |




A Julia package for simulating high-dimensional multivariate data with a target correlation and arbitrary marginal distributions via Gaussian copula. *Bigsimr* works with any univariate distribution implemented in *Distributions.jl* or any user-defined distribution derived from *Distributions* univariate classes. Additionally, *Bigsimr* accounts for different target correlations:

- Pearson: employs a matching algorithm (Xiao and Zhou 2019) to account for the non-linear transformation in the Normal-to-Anything (NORTA) step
- Spearman and Kendall: Use explicit transformations (Lebrun and Dutfoy 2009)

## Other Features

* **Nearest Correlation Matrix** - Calculate the nearest positive [semi]definite correlation matrix (Qi and Sun 2006)
* **Fast Approximate Correlation Matrix** - Calculate an approximation to the nearest positive definite correlation matrix
* **Random Correlation Matrix** - Generate random positive [semi]definite correlation matrices 
* **Fast Multivariate Normal Generation** - Utilize multithreading to generate multivariate normal samples in parallel

## Examples

Pearson matching

```julia
using Bigsimr
using Distributions

target_corr = cor_randPD(3)
margins = [Binomial(20, 0.2), Beta(2, 3), LogNormal(3, 1)]

adjusted_corr = pearson_match(target_corr, margins)

x = rvec(100_000, adjusted_corr, margins)
cor(x, Pearson)
```

Spearman/Kendall matching

```julia
spearman_corr = cor_randPD(3)
adjusted_corr = cor_convert(spearman_corr, Spearman, Pearson)

x = rvec(100_000, adjusted_corr, margins)
cor(x, Spearman)
```

Nearest correlation matrix

```julia
import LinearAlgebra: isposdef

s = cor_randPSD(200)
r = cor_convert(s, Spearman, Pearson)
isposdef(r)

p = cor_nearPD(r)
isposdef(p)
```

Fast approximate nearest correlation matrix

```julia
s = cor_randPSD(2000)
r = cor_convert(s, Spearman, Pearson)
isposdef(r)

p = cor_fastPD(r)
isposdef(p)
```

## References

* Xiao, Q., & Zhou, S. (2019). Matching a correlation coefficient by a Gaussian copula. Communications in Statistics-Theory and Methods, 48(7), 1728-1747.
* Lebrun, R., & Dutfoy, A. (2009). An innovating analysis of the Nataf transformation from the copula viewpoint. Probabilistic Engineering Mechanics, 24(3), 312-320.
* Qi, H., & Sun, D. (2006). A quadratically convergent Newton method for computing the nearest correlation matrix. SIAM journal on matrix analysis and applications, 28(2), 360-385.
* amoeba (https://stats.stackexchange.com/users/28666/amoeba), How to generate a large full-rank random correlation matrix with some strong correlations present?, URL (version: 2017-04-13): https://stats.stackexchange.com/q/125020



[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://adknudson.github.io/Bigsimr.jl/stable

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://adknudson.github.io/Bigsimr.jl/dev

[codecov-img]: https://codecov.io/gh/adknudson/Bigsimr.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/adknudson/Bigsimr.jl

[ci-img]: https://github.com/adknudson/Bigsimr.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/adknudson/Bigsimr.jl/actions/workflows/CI.yml

[release-img]: https://img.shields.io/github/v/tag/adknudson/Bigsimr.jl?label=release&sort=semver

[license-img]: https://img.shields.io/github/license/adknudson/Bigsimr.jl
[license-url]: https://choosealicense.com/licenses/mit/

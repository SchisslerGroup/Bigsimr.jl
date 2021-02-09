# bigsimr

| **Documentation**                       | **Build and Coverage**                    | **Package Details**                    |
|:---------------------------------------:|:-----------------------------------------:|:--------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] | [![Build Status][travis-img]][travis-url] | [![Licence][license-img]][license-url] |
| [![][docs-latest-img]][docs-latest-url] | [![Coverage][codecov-img]][codecov-url]   | ![Release][release-img]                |


A Julia package for simulating high-dimensional multivariate data with a target correlation and arbitrary marginal distributions. *bigsimr* works with any distribution implemented in *Distributions.jl* or any user-defined distribution derived from *Distributions* univariate classes. Additionally, *bigsimr* accounts for different target correlations:

- Pearson: employs a matching algorithm (Xioa and Zhou 2019) to account for the non-linear transformation in the Normal-To-Anything (NORTA) step
- Spearman and Kendall: Use explicit transformations (Avramidis et al. 2009, Lebrun and Dutfoy 2009) and calculate the nearest positive definite correlation matrix (Qi and Sun 2006) before doing the NORTA step


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://adknudson.github.io/bigsimr.jl/stable

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://adknudson.github.io/bigsimr.jl/dev

[travis-img]: https://travis-ci.com/adknudson/bigsimr.jl.svg?branch=master
[travis-url]: https://travis-ci.com/adknudson/bigsimr.jl

[codecov-img]: https://codecov.io/gh/adknudson/bigsimr.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/adknudson/bigsimr.jl

[release-img]: https://img.shields.io/github/v/tag/adknudson/bigsimr.jl?label=release&sort=semver

[license-img]: https://img.shields.io/github/license/adknudson/bigsimr.jl
[license-url]: https://choosealicense.com/licenses/mit/

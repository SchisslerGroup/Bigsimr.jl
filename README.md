# MvSim

| **Documentation**                       | **Build and Coverage**                    | **Package Details**                    |
|:---------------------------------------:|:-----------------------------------------:|:--------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] | [![Build Status][travis-img]][travis-url] | [![Licence][license-img]][license-url] |
| [![][docs-latest-img]][docs-latest-url] | [![Coverage][codecov-img]][codecov-url]   | ![Release][release-img]                |


A Julia package for simulating high-dimensional multivariate data with a target correlation and arbitrary marginal distributions. *MvSim* works with any distribution implemented in *Distributions.jl* or any user-defined distribution derived from *Distributions* univariate classes. Additionally, *MvSim* accounts for different target correlations:

- Pearson: employs a matching algorithm (Xioa and Zhou 2019) to account for the non-linear transformation in the Normal-To-Anything (NORTA) step
- Spearman and Kendall: Use explicit transformations (Avramidis et al. 2009, Lebrun and Dutfoy 2009) and calculate the nearest positive definite correlation matrix (Qi and Sun 2006) before doing the NORTA step


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://adknudson.github.io/MvSim.jl/stable

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://adknudson.github.io/MvSim.jl/dev

[travis-img]: https://travis-ci.com/adknudson/MvSim.jl.svg?branch=master
[travis-url]: https://travis-ci.com/adknudson/MvSim.jl

[codecov-img]: https://codecov.io/gh/adknudson/MvSim.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/adknudson/MvSim.jl

[release-img]: https://img.shields.io/github/v/tag/adknudson/MvSim.jl?label=release&sort=semver

[license-img]: https://img.shields.io/github/license/adknudson/MvSim.jl
[license-url]: https://choosealicense.com/licenses/mit/

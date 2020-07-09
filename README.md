# MvSim

[![Build Status](https://travis-ci.com/adknudson/MvSim.jl.svg?branch=master)](https://travis-ci.com/adknudson/MvSim.jl)
[![Coverage](https://codecov.io/gh/adknudson/MvSim.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/adknudson/MvSim.jl)
![License](https://img.shields.io/github/license/adknudson/MvSim.jl)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adknudson.github.io/MvSim.jl//stable)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adknudson.github.io/MvSim.jl//dev)
![Release](https://img.shields.io/github/v/tag/adknudson/MvSim.jl?label=release&sort=semver)

A Julia package for simulating high-dimensional multivariate data with a target correlation and arbitrary marginal distributions. *MvSim* works with any distribution implemented in *Distributions.jl* or any user-defined distribution derived from *Distributions* univariate classes. Additionally, *MvSim* accounts for different target correlations:

- Pearson: employs a matching algorithm (Xioa and Zhou 2019) to account for the non-linear transformation in the Normal-To-Anything (NORTA) step
- Spearman and Kendall: Use explicit transformations (Avramidis et al. 2009, Lebrun and Dutfoy 2009) and calculate the nearest positive semidefinite correlation matrix (Qi and Sun 2006) before doing the NORTA step

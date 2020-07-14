module MvSim

using Distributions
using IntervalArithmetic
using Match

import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots
import LinearAlgebra: diagind, diagm, diag, eigen, norm, pinv, I
import Memoize: @memoize
import Polynomials: Polynomial
import Statistics: mean, std, quantile, clampcor

const P = "pearson"
const S = "spearman"
const K = "kendall"

export
    nearestPSDcor,
    ρz,
    ρz_bounds,

    # Types
    MixedMultivariateDistribution,

    # utilities
    cor2cor,
    cov2cor,
    get_coefs,
    hermite,
    rcor

include("utils.jl")
include("MixedMultivariateDistribution.jl")
include("nearestPSDcor.jl")
include("PearsonMatching.jl")

end

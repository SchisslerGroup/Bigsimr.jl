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
import Base: promote

const CorrelationTypes = Dict(
    :P => "Pearson",
    :S => "Spearman",
    :K => "Kendall")

export
    nearestPSDcor,
    ρz,
    ρz_bounds,

    # Types
    MixedMultivariateDistribution,
    CorrelationTypes,

    # utilities
    cor2cor,
    cov2cor,
    hermite,
    rcor,

    # Extended Base utilities
    promote

include("utils.jl")
include("MixedMultivariateDistribution.jl")
include("nearestPSDcor.jl")
include("PearsonMatching.jl")

end

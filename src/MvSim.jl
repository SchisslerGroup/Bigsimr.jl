module MvSim

using Distributions
using IntervalArithmetic
using Match

import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots
import LinearAlgebra: diagind, diagm, diag, eigen, norm2, pinv, I
import Memoize: @memoize
import Polynomials: Polynomial
import Statistics: mean, std, quantile, cor, cov2cor!, clampcor
import StatsBase: corspearman, corkendall
import Base: promote

abstract type AbstractCorrelation end
abstract type Correlation <: AbstractCorrelation end
struct Pearson  <: Correlation end
struct Spearman <: Correlation end
struct Kendall  <: Correlation end

export
    cor_nearPSD,
    cor_randPSD,
    cor_randPD,
    
    # Pearson correlation matching
    ρz,
    ρz_bounds,

    # utilities
    hermite,

    # Extended Base utilities
    promote

include("utils.jl")
include("cor_nearPSD.jl")
include("cor_rand.jl")
include("cor_utils.jl")
include("PearsonMatching.jl")

end

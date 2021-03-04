module Bigsimr

import Base.Threads: @threads
import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution, ContinuousUnivariateDistribution
import Distributions: mean, std, quantile, cdf
import FastGaussQuadrature: gausshermite
import IntervalArithmetic: interval, mid
import IntervalRootFinding: roots, Krawczyk
import IterTools: subsets
import LinearAlgebra: diagind, diagm, diag, Diagonal,
                      eigen, norm, inv, I, Symmetric,
                      cholesky, isposdef, issymmetric
import Polynomials: Polynomial, derivative
import SharedArrays: SharedMatrix, sdata
import SpecialFunctions: erfc, erfcinv
import Statistics: cor, clampcor
import StatsBase: corspearman, corkendall
import GeneralizedSDistributions: GSDist


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

struct ValidCorrelationError <: Exception end

const sqrt2 = sqrt(2)
const invsqrt2 = inv(sqrt(2))
const invsqrtpi = inv(sqrt(π))
const invsqrt2π = inv(sqrt(2π))


abstract type Correlation end
"""
    Pearson <: Correlation

Pearson's ``r`` product-moment correlation
"""
struct Pearson <: Correlation end
"""
    Spearman <: Correlation

Spearman's ``ρ`` rank correlation
"""
struct Spearman <: Correlation end
"""
    Kendall <: Correlation

Kendall's ``τ`` rank correlation
"""
struct Kendall <: Correlation end


export
    rvec, rmvn,
    # Pearson matching
    pearson_match, pearson_bounds,
    # Correlation Types
    Correlation, Pearson, Spearman, Kendall,
    # Correlation Utils
    cor, cor_fast,
    cor_nearPD, cor_fastPD, cor_fastPD!,
    cor_randPD, cor_randPSD,
    cor_convert,
    cor_bounds,
    cor_constrain, cor_constrain!,
    cov2cor, cov2cor!,
    clamp, cor_clamp


include("utils.jl")

include("RandomVector/rvec.jl")
include("RandomVector/rmvn.jl")
include("RandomVector/utils.jl")

include("Correlation/cor_bounds.jl")
include("Correlation/fast_pos_def.jl")
include("Correlation/nearest_pos_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")


include("precompile.jl")


end

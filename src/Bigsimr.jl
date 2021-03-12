module Bigsimr

using Distributions, LinearAlgebra
using PDMats
using SharedArrays

using Base.Threads: @threads

using FastGaussQuadrature: gausshermite
using HypergeometricFunctions: _₂F₁
using IntervalArithmetic: interval, mid
using IntervalRootFinding: roots, Krawczyk
using IterTools: subsets
using LsqFit: curve_fit, coef
using Polynomials: Polynomial, derivative
using QuadGK: quadgk
using SpecialFunctions: erfc, erfcinv
using StatsBase: corspearman, corkendall


import Distributions: mean, std, quantile, cdf, pdf, var, params
import LinearAlgebra: diag, inv, logdet
import PDMats: dim,
               quad, quad!, invquad!, invquad,
               pdadd, pdadd!,
               X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X,
               whiten!, unwhiten!
import Statistics: cor, clampcor


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
    PDCorMat,
    # Correlation Utils
    cor, cor_fast,
    cor_nearPD, cor_fastPD, cor_fastPD!,
    cor_randPD, cor_randPSD,
    cor_convert,
    cor_bounds,
    cor_constrain, cor_constrain!,
    cov2cor, cov2cor!,
    clamp, cor_clamp,
    iscorrelation


include("utils.jl")

include("PDCorMat.jl")

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

include("GSDist/GSDist.jl")
include("GSDist/utils.jl")

include("precompile.jl")


end

module Bigsimr

using Base.Threads: @threads
using Distributions
using FastGaussQuadrature: gausshermite
using HypergeometricFunctions: _₂F₁
using IrrationalConstants
using IntervalArithmetic: interval, mid
using IntervalRootFinding: roots, Krawczyk
using IterTools: subsets
using LinearAlgebra
using LsqFit: curve_fit, coef
using Polynomials: Polynomial, derivative
using QuadGK: quadgk
using SharedArrays
using SpecialFunctions: erfc, erfcinv
using StatsBase: corspearman, corkendall
using Statistics: clampcor

import Distributions: mean, std, quantile, cdf, pdf, var, params
import Statistics


struct ValidCorrelationError <: Exception end


"""
    CorType

A type used for specifiying the type of correlation. Supported correlations are:

    - [`Pearson`](@ref)
    - [`Spearman`](@ref)
    - [`Kendall`](@ref)
"""
struct CorType{T} end

"""
    Pearson

Pearson's ``r`` product-moment correlation
"""
const Pearson = CorType{:Pearson}()

"""
    Spearman

Spearman's ``ρ`` rank correlation
"""
const Spearman = CorType{:Spearman}()

"""
    Kendall

Kendall's ``τ`` rank correlation
"""
const Kendall = CorType{:Kendall}()



const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution


include("utils.jl")
include("random_vector.jl")

include("cor_bounds.jl")
include("cor_fastPD.jl")
include("cor_nearPD.jl")
include("cor_random.jl")
include("cor_utils.jl")

include("pearson_utils.jl")
include("pearson_match.jl")
include("pearson_bounds.jl")

include("GSDist.jl")



export
    # correlation types
    CorType,
    Pearson,
    Spearman,
    Kendall,
    # random vector generation
    rvec,
    rmvn,
    # correlation calculation
    cor,
    cor_fast,
    cor_convert,
    cor_bounds,
    # nearest correlation
    cor_nearPD,
    cor_fastPD,
    cor_fastPD!,
    # random correlation generation
    cor_randPD,
    cor_randPSD,
    # correlation Utils
    iscorrelation,
    cor_constrain,
    cor_constrain!,
    cov2cor,
    cov2cor!,
    # pearson methods
    pearson_match,
    pearson_bounds



end

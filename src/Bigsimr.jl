module Bigsimr

using Base.Threads: @threads
using Distributions
using FastGaussQuadrature: gausshermite
using HypergeometricFunctions: _₂F₁
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

import Distributions: mean, std, quantile, cdf, pdf, var, params
import LinearAlgebra: diag, inv, logdet
import Statistics: cor, clampcor


struct ValidCorrelationError <: Exception end


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


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

const sqrt2_f64      ::Float64 = sqrt(2)
const invsqrt2_f64   ::Float64 = inv(sqrt(2))
const invsqrtpi_f64  ::Float64 = inv(sqrt(π))
const invsqrt2pi_f64 ::Float64 = inv(sqrt(2π))

const sqrt2_f32      ::Float32 = sqrt2_f64
const invsqrt2_f32   ::Float32 = invsqrt2_f64
const invsqrtpi_f32  ::Float32 = invsqrtpi_f64
const invsqrt2pi_f32 ::Float32 = invsqrt2pi_f64



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
    Correlation, 
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

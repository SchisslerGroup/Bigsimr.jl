module Bigsimr

using Distributions
using IntervalArithmetic

import Base: promote, rand, eltype
import Base.Threads: @threads
import FastGaussQuadrature: gausshermite
import HypergeometricFunctions: _₂F₁
import IntervalRootFinding: roots, Krawczyk
import IterTools: subsets
import LinearAlgebra: diagind, diagm, diag, Diagonal,
                      eigen, norm, inv, I, Symmetric,
                      cholesky, isposdef, issymmetric
import LsqFit: curve_fit, coef
import Polynomials: Polynomial, derivative
import QuadGK: quadgk
import SharedArrays: SharedMatrix, sdata
import SpecialFunctions: erfc, erfcinv
import Statistics: cor, clampcor
import StatsBase: corspearman, corkendall


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

struct ValidCorrelationError <: Exception end
function iscorrelation(X::Matrix{<:AbstractFloat})
    all([
        isposdef(X),
        issymmetric(X),
        all(diag(X) .== one(eltype(X))),
        all(-one(eltype(X)) .≤ X .≤ one(eltype(X)))
    ])
end

const sqrt2 = sqrt(2)
const invsqrt2 = inv(sqrt(2))
const invsqrtpi = inv(sqrt(π))
const invsqrt2π = inv(sqrt(2π))

_normpdf(x::Float64) = exp(-abs2(x)/2) * invsqrt2π
_normcdf(x::Float64) = erfc(-x * invsqrt2) / 2
_norminvcdf(x::Float64) = -√(2) * erfcinv(2x)


export
rvec, rmvn,
MvDistribution, margins, cortype,
GSDistribution, quantile, mean, var, std,
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
cov2cor, cov2cor!
# Extended Base utilities
promote,
rand,
eltype


include("MvDistribution.jl")
include("rand_vec.jl")
include("GSDistribution.jl")

include("Correlation/nearest_pos_def.jl")
include("Correlation/fast_pos_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")

end

module Bigsimr

using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution
using LinearAlgebra: issymmetric, isposdef, cholesky, Diagonal, diagm, diag
using SharedArrays
using Statistics
using StatsBase: corspearman, corkendall
using StatsFuns: normcdf
using NearestCorrelationMatrix


export
    # Correlation Types
    CorType,
    Pearson,
    Spearman,
    Kendall,
    # Correlation Methods
    cor,
    cor_fast,
    cor_convert,
    cor_constrain,
    cor_constrain!,
    cov2cor,
    cov2cor!,
    is_correlation,
    # Random Correlation Generation
    cor_randPSD,
    cor_randPD,
    # Correlation Bounds
    cor_bounds,
    # Nearest Correlation Matrix
    cor_nearPD,
    cor_nearPD!,
    cor_nearPSD,
    cor_nearPSD!,
    cor_fastPD,
    cor_fastPD!,
    # Random Vector Generation
    rmvn,
    rvec


using Reexport
@reexport using Distributions
@reexport using PearsonCorrelationMatch
@reexport using NearestCorrelationMatrix


# shorthand constants
const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution


"""
    is_correlation(X)

Check if the given matrix passes all the checks required to be a valid correlation matrix.
"""
function is_correlation(X::AbstractMatrix{T}) where {T<:Real}
    issymmetric(X)     || return false
    all(diag(X) .== 1) || return false
    all(-1 .≤ X .≤ 1)  || return false
    isposdef(X)        || return false

    return true
end


include("common.jl")
include("correlations.jl")
include("rand.jl")

end

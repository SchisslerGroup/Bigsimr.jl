module Bigsimr

using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution
using LinearAlgebra: issymmetric, isposdef, cholesky, Diagonal, diagm, diag
using SharedArrays
using Statistics
using Statistics: cor
using StatsBase: corspearman, corkendall
using StatsFuns: normcdf
using PearsonCorrelationMatch
using PearsonCorrelationMatch: pearson_match


export
    # Correlation Types
    CorType,
    Pearson,
    Spearman,
    Kendall,
    # Correlation Methods
    cor,
    cor_fast,
    cor_bounds,
    cor_convert,
    cor_constrain,
    cor_constrain!,
    cov2cor,
    cov2cor!,
    is_correlation,
    # Random Correlation Generation
    cor_randPSD,
    cor_randPD,
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


function PearsonCorrelationMatch.pearson_match(rho::AbstractMatrix{T}, margins::Vector{<:UD}) where {T<:Real}
    d = length(margins)
    r, s = size(rho)
    (r == s == d) || throw(DimensionMismatch("The number of margins must be the same size as the correlation matrix."))

    R = SharedMatrix{Float64}(d, d)

    # Calculate the pearson matching pairs
    Base.Threads.@threads for (i, j) in _idx_subsets2(d)
        @inbounds R[i, j] = pearson_match(rho[i,j], margins[i], margins[j])
    end

    _symmetric!(R)
    _set_diag1!(R)
    return cor_fastPD!(R)
end


include("common.jl")
include("correlations.jl")
include("rand.jl")

end

module Bigsimr

using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution
using LinearAlgebra: issymmetric, isposdef, cholesky, Diagonal, diagm, diag
using NearestCorrelationMatrix
using PearsonCorrelationMatch
using PearsonCorrelationMatch: pearson_match, pearson_bounds
using SharedArrays
using Statistics
using Statistics: cor
using StatsBase: corspearman, corkendall
using StatsFuns: normcdf


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


function __init__()
    if :Distributions ∉ names(Main; imported=true)
        @info """

          Bigsimr.jl gains a lot of functionality
          when used with Distributions.jl, which
          is not currently loaded. If you have it
          installed, then you can load it by:

        julia> using Distributions

          If you don't have Distributions.jl installed,
          then you can add it with:

        julia> using Pkg; Pkg.install("Distributions")

        julia> using Distributions

        """
    end
end

end

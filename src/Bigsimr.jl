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

# Criteria

A matrix is a valid correlation matrix if:

- Square
- Symmetric
- Diagonal elements are equal to `1`
- Off diagonal elements are between `-1` and `1`
- Is positive definite

# Examples

```julia-repl
julia> x = rand(3, 3)
3×3 Matrix{Float64}:
 0.834446  0.183285  0.837872
 0.637295  0.270709  0.458703
 0.626566  0.736907  0.61903

julia> is_correlation(x)
false

julia> x = cor_randPD(3)
3×3 Matrix{Float64}:
 1.0       0.190911  0.449104
 0.190911  1.0       0.636305
 0.449104  0.636305  1.0

julia> is_correlation(x)
true

julia> r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
];

julia> is_correlation(r_negdef)
false
```
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

        """
    end
end

end

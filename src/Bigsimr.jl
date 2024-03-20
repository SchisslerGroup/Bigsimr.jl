module Bigsimr


using Distributions: UnivariateDistribution as UD
using Distributions: ContinuousUnivariateDistribution as CUD
using Distributions: DiscreteUnivariateDistribution as DUD

using LinearAlgebra: issymmetric, isposdef, cholesky, Diagonal, diagm, diag
using NearestCorrelationMatrix
using NearestCorrelationMatrix.Internals: isprecorrelation
using SharedArrays
using Statistics: cor
using StatsBase: corspearman, corkendall

import Statistics


using Reexport
@reexport using PearsonCorrelationMatch: pearson_bounds, pearson_match
@reexport using NearestCorrelationMatrix: nearest_cor, nearest_cor!
@reexport using NearestCorrelationMatrix: Newton, AlternatingProjections, DirectProjection


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
    cor_nearPSD,
    cor_nearPSD!,
    cor_nearPD,
    cor_nearPD!,
    cor_fastPD,
    cor_fastPD!,
    # Random Vector Generation
    rmvn,
    rvec


include("internals/Internals.jl")
using .Internals

include("cortype.jl")
include("cor.jl")
include("cor_utils.jl")
include("cor_gen.jl")
include("nearest_cor.jl")
include("rand.jl")


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
is_correlation(X) = isprecorrelation(X) && isposdef(X)


end

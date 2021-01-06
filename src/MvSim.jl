module MvSim

using Distributions
using IntervalArithmetic

import Base: promote, rand, eltype
import Base.Threads: @threads
import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots, Krawczyk
import IterTools: subsets
import LinearAlgebra: diagind, diagm, diag, Diagonal,
                      eigen, norm, inv, I, Symmetric,
                      cholesky, isposdef, issymmetric
import Memoize: @memoize
import Polynomials: Polynomial, derivative
import SharedArrays: SharedMatrix, sdata
import Statistics: mean, std, quantile, cor, clampcor
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


export
rvec, 
MvDistribution, margins, cortype,
# Pearson matching
pearson_match, pearson_bounds,
# Correlation Types
Correlation, Pearson, Spearman, Kendall,
# Correlation Utils
cor,
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
include("hermite.jl")

include("Correlation/nearest_pos_def.jl")
include("Correlation/fast_pos_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")

end

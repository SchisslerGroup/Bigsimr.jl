module MvSim

using Distributions
using IntervalArithmetic

import Base: promote, rand
import Base.Threads: @threads
import Match: @match
import Memoize: @memoize
import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots
import LinearAlgebra: diagind, diagm, diag, eigen, norm2, pinv, I, Symmetric
import Polynomials: Polynomial
import Statistics: mean, std, quantile, cor, clampcor
import StatsBase: corspearman, corkendall


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

abstract type Correlation end
struct Pearson  <: Correlation end
struct Spearman <: Correlation end
struct Kendall  <: Correlation end

"""
    MvDistribution(R, margins, C)

Simple data structure for storing a multivariate mixed distribution.
"""
struct MvDistribution
    R::Matrix{<:Real}
    margins::Vector{<:UD}
    C::Type{<:Correlation}
end


export
rvec, MvDistribution,
# Pearson matching
ρz, ρz_bounds,
# Correlation Types
Correlation, Pearson, Spearman, Kendall,
# Correlation Utils
cor,
cor_nearPD,
cor_nearPSD,
cor_randPD,
cor_randPSD,
cor_convert,
# Extended Base utilities
promote,
rand


include("rand_vec.jl")
include("hermite.jl")
include("utils.jl")

include("Correlation/nearest_pos_def.jl")
include("Correlation/nearest_pos_semi_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")

include("Parallel/Parallel.jl")

end

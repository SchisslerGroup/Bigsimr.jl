module MvSim

using Distributions
using IntervalArithmetic

import Base: promote, rand, eltype
import Base.Threads: @threads
import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots
import IterTools: subsets
import LinearAlgebra: diagind, diagm, diag, eigen, norm, inv, I, Symmetric, 
                      cholesky
import Match: @match
import Memoize: @memoize
import Polynomials: Polynomial
import SharedArrays: SharedMatrix, sdata
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
    ρ::Matrix{<:Real}
    F::Vector{<:UD}
    C::Type{<:Correlation}
end
margins(D::MvDistribution) = D.F
cor(D::MvDistribution)     = D.ρ
cortype(D::MvDistribution) = D.C
eltype(D::MvDistribution)  = eltype(D.ρ)


export
rvec, 
MvDistribution, margins, cortype,
# Pearson matching
pearson_match, pearson_bounds,
# Correlation Types
Correlation, Pearson, Spearman, Kendall,
# Correlation Utils
cor,
cor_nearPD,
cor_randPD, cor_randPSD,
cor_convert,
cor_bounds,
# Extended Base utilities
promote,
rand,
eltype


include("rand_vec.jl")
include("hermite.jl")

include("Correlation/nearest_pos_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")

end

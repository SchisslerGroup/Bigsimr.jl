module MvSim

using Distributions
using IntervalArithmetic

import Base: promote
import Base.Threads: @threads
import Match: @match
import Memoize: @memoize
import FastGaussQuadrature: gausshermite
import IntervalRootFinding: roots
import LinearAlgebra: diagind, diagm, diag, eigen, norm2, pinv, I
import Polynomials: Polynomial
import Statistics: mean, std, quantile, cor, clampcor
import StatsBase: corspearman, corkendall

const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

struct MvDistribution
    R::Matrix{<:Real}
    margins::Vector{<:UD}
    C::Type{<:Correlation}
end

export
rvec, MvDistribution,
# utilities
hermite,
# Extended Base utilities
promote

include("rand_vec.jl")
include("hermite.jl")
include("utils.jl")

include("Correlation/Correlation.jl")
include("PearsonMatching/PearsonMatching.jl")
include("Parallel/Parallel.jl")

end

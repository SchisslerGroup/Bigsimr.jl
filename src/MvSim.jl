module MvSim

using Distributions
using Match

import LinearAlgebra: diagind, diagm, diag, eigen, norm, pinv, I
import Memoize: @memoize

export
    nearestPSDcor,

    # utilities
    cor2cor,
    cov2cor,
    hermite,
    rcor

include("utils.jl")
include("nearestPSDcor.jl")

end

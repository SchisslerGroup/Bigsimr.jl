module MvSim

using Distributions

import LinearAlgebra: diagind, diagm, diag, eigen, norm, pinv, I

include("nearestPSDcor.jl")
include("utils.jl")

export
    cov2cor,
    rcor,
    nearestPSDcor

end

module MvSim

using Distributions

import LinearAlgebra: diagind, diagm, diag, eigen, norm, pinv, I

export
    cov2cor,
    rcor,
    nearestPSDcor

include("utils.jl")
include("nearestPSDcor.jl")

end

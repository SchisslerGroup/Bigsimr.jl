module MvSim

import LinearAlgebra: diagind, diagm, diag, eigen, norm, pinv, I

include("nearestSPDcor.jl")
include("utils.jl")

export
    cov2cor,
    rcor,
    nearestSPDcor

end

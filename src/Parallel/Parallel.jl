module Parallel

using Distributions
using Distributed: @distributed, @everywhere
using SharedArrays: SharedMatrix, sdata

function rvec(n::Int, margins::Vector{<:UnivariateDistribution}, ρ::Matrix{<:Real})
    @everywhere margins = $margins
    d = length(margins)
    Z = rand(MvNormal(ρ), n)'
    X = SharedMatrix(similar(Z))
    @sync @distributed for i in 1:d
        @inbounds X[:,i] = normal_to_margin(margins[i], Z[:,i])
    end
    return sdata(X)
end

end
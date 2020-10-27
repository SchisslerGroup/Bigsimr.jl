"""
    rvec(n, margins, ρ)
"""
function rvec end

function rvec(n::Int, margins::Vector{<:UD}, ρ::Matrix{<:Real})
    d = length(margins)
    Z = rand(MvNormal(ρ), n)'
    X = similar(Z)
    @threads for i in 1:d
        @inbounds X[:,i] = normal_to_margin(margins[i], Z[:,i])
    end
    return X
end
"""
    rmvn(n[, μ], Σ)

Fast parrallel generation of multivariate normal samples.
"""
function rmvn end
for T in (Float64, Float32, Float16)
    @eval function rmvn(n::Int, μ::Vector{$T}, Σ::Matrix{$T})
        μ' .+ _rmvn(n, Σ)
    end
    @eval rmvn(n::Real, μ::Vector{$T}, Σ::Matrix{$T}) = rmvn(Int(n), μ, Σ)
    @eval function rmvn(n::Real, Σ::Matrix{$T})
        d = size(Σ, 2)
        rmvn(Int(n), zeros($T, d), Σ)
    end
end
function rmvn(n::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real})
    rmvn(Int(n), Vector{Float64}(μ), Matrix{Float64}(Σ))
end

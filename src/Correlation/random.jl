"""
    cor_randPSD(T::Type{<:AbstractFloat}, d::Int, k::Int=d)

Compute a random positive semidefinite correlation matrix

Reference
- https://stats.stackexchange.com/a/125020
- https://www.sciencedirect.com/science/article/pii/S0047259X09000876
"""
function cor_randPSD end

function cor_randPSD(T::Type{<:AbstractFloat}, d::Int, k::Int=d)
    @assert d ≥ 1
    @assert 1 ≤ k ≤ d

    d == 1 && return ones(T, 1, 1)

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    R = S2 * S * S2

    cor_constrain(R)
end
cor_randPSD(d::Int, k::Int=d) = cor_randPSD(Float64, d, k)

cor_randPD(T::Type{<:AbstractFloat}, d::Int, k::Int=d) = cor_nearPD(cor_randPSD(T, d, k))
cor_randPD(d::Int, k::Int=d) = cor_nearPD(cor_randPSD(d, k))
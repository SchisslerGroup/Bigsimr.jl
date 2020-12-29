function npsd_getAplus(A::Matrix{T}) where {T<:Real}
    λ, P = eigen(A)
    λ .= max.(λ, zero(eltype(λ)))
    P * diagm(λ) * P'
end

function npsd_getPs(A::Matrix{T}, W::Matrix{T}) where {T<:Real}
    W½ = W.^0.5
    pinv(W½) * npsd_getAplus(W½ * A * W½) * pinv(W½)
end

function npsd_getPu(A::Matrix{T}, W::Matrix{T}) where {T<:Real}
    B = copy(A)
    B[W .> 0] .= W[W .> 0]
    B
end


"""
    cor_nearPSD(A::Matrix{T}; n_iter::Int=100) where {T<:Real}

# Examples
```julia
import LinearAlgebra: eigvals
# Define a negative definite correlation matrix
ρ = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]
eigvals(ρ)

r = cor_nearPSD(ρ, n_iter=100)
eigvals(r)
```
"""
function cor_nearPSD(A::Matrix{T}; n_iter::Int=3) where {T<:Real}
    n = size(A, 1)
    W = Matrix{T}(I, size(A))
    δₛ = zeros(T, size(A))
    Yₖ = copy(A)
    for k ∈ 1:n_iter
        Rₖ = Yₖ - δₛ
        Xₖ = npsd_getPs(Rₖ, W)
        δₛ = Xₖ - Rₖ
        Yₖ = npsd_getPu(Xₖ, W)
    end

    cor_constrain(Yₖ)
end

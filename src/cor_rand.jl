"""
    cor_randPD

Generate a random positive definite correlation matrix of size ``d×d``. The
parameter `α` is used to determine the autocorrelation in the correlation
coefficients.

Reference
- Joe H (2006). Generating random correlation matrices based on partial
  correlations. J. Mult. Anal. Vol. 97, 2177--2189.
"""
function cor_randPD end

function _rjm(T::Type, A, a)
    b     = size(A, 1)
    idx   = 2 ≤ b-1 ? range(2, b-1, step=1) : range(2, b-1, step=-1)
    ρ₁    = A[idx, 1]
    ρ₃    = A[idx, b]
    R₂    = A[idx, idx]
    Rᵢ    = pinv(R₂)
    rcond = 2 * T(rand(Beta(a, a))) - 1
    t13   = ρ₁' * Rᵢ * ρ₃
    t11   = ρ₁' * Rᵢ * ρ₁
    t33   = ρ₃' * Rᵢ * ρ₃
    return t13[1] + rcond * √((1 - t11[1]) * (1 - t33[1]))
end

function cor_randPD(::Type{T}, d::Int, α::Real=1.0) where {T<:AbstractFloat}
    if d == 1
        return ones(T, 1, 1)
    elseif d == 2
        ρ = 2*rand(T) - 1
        return Matrix{T}([1 ρ; ρ 1])
    else
        R = Matrix{T}(I, d, d)

        for i=1:d-1
            α₀ = α + (d - 2) / 2
            R[i, i+1] = 2 * T(rand(Beta(α₀, α₀))) - 1
            R[i+1, i] = R[i, i+1]
        end

        for m = 2:d-1, j = 1:d-m
            r_sub = R[j:j+m, j:j+m]
            α₀ = α + (d - m - 1) / 2
            R[j, j+m] = _rjm(T, r_sub, α₀)
            R[j+m, j] = R[j, j+m]
        end

        return R
    end
end
cor_randPD(           d::Int, α::Real)                          = cor_randPD(Float64, d, α)
cor_randPD(::Type{T}, d::Int         ) where {T<:AbstractFloat} = cor_randPD(T,       d, 1)
cor_randPD(           d::Int         )                          = cor_randPD(Float64, d, 1)


"""
    cor_randPSD()

Compute a random positive semidefinite correlation matrix
"""
function cor_randPSD end

function cor_randPSD(::Type{T}, d::Int, k::Int) where {T<:AbstractFloat}
    if d == 1
        return ones(T, 1, 1)
    end

    @assert d ≥ 1
    @assert 1 ≤ k ≤ d

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    clampcor.(S2 * S * S2)
end
cor_randSPD(           d::Int, k::Int)                          = cor_randPD(Float64, d, k)
cor_randSPD(::Type{T}, d::Int        ) where {T<:AbstractFloat} = cor_randPD(T,       d, 1)
cor_randSPD(           d::Int        )                          = cor_randPD(Float64, d, 1)
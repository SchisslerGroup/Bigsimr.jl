"""
    cor2cor(ρ::T, from::String, to::String) where {T <: Real}

Convert from one type of correlation matrix to another. The possible correlation
types are _pearson_, _spearman_, or _kendall_. If an invalid pair is given, return
the original correlation matrix.
"""
function cor2cor(ρ::T, from::String, to::String) where {T <: Real}
    from, to = lowercase(from), lowercase(to)

    if from == to
        return ρ
    end

    @match (from, to) begin
        ("pearson", "spearman") => (6 / π) .* asin.(ρ ./ 2)
        ("pearson", "kendall")  => (2 / π) .* asin.(ρ)
        ("spearman", "pearson") => 2 * sin.(ρ .* π / 6)
        ("spearman", "kendall") => (2 / π) .* asin.(2 * sin.(ρ .* π ./ 6))
        ("kendall", "pearson")  => sin.(ρ .* π / 2)
        ("kendall", "spearman") => (6 / π) .* asin.(sin.(ρ .* π ./ 2) ./ 2)
        (f, t) => throw(ArgumentError("No matching conversion from '$f' to '$t'. 'from' and 'to' must be any combination of 'pearson', 'spearman', or 'kendall'"))
    end
end

function cor2cor(A::AbstractArray{T}, from::String, to::String) where {T <: Real}
    cor2cor.(A, from, to)
end


"""
    cov2cor(Σ::AbstractArray)

Convert a covariance matrix to a correlation matrix. Ensure that the resulting
matrix is symmetric and has diagonals equal to 1.0.
"""
function cov2cor(Σ::AbstractArray)
    D = pinv(diagm(sqrt.(diag(Σ))))
    D .= D * Σ * D
    setdiag!(D, 1.0)
    (D + D') / 2
end


"""
    hermite(x, n::Int, probabilists::Bool=true)

Compute the Hermite polynomials of degree `n`. Compute the Probabilists' version
by default.

The two definitions of the Hermite polynomials are each a rescaling of the other.
Let ``Heₙ(x)`` denote the Probabilists' version, and ``Hₙ(x)`` the Physicists'.
Then

```math
H_{n}(x) = 2^{\\frac{n}{2}} He_{n}\\left(\\sqrt{2} x\\right)
```

```math
He_{n}(x) = 2^{-\\frac{n}{2}} H_{n}\\left(\\frac{x}{\\sqrt{2}}\\right)
```
"""
function hermite(x, n::Int, probabilists::Bool=true)
    @memoize function _h(x, n)
        if n == 0
            return length(x) > 1 ? ones(length(x)) : 1
        elseif n == 1
            return x
        else
            return x .* _h(x, n-1) .- (n-1) .* _h(x, n-2)
        end
    end

    if probabilists
        return _h(x, n)
    else
        return 2^(n/2) * _h(x*√2, n)
    end
end


"""
    rcor(d::Integer, α::Real=1.0)

Generate a random positive definite correlation matrix of size ``d×d``. The
parameter `α` is used to determine the autocorrelation in the correlation
coefficients.

Reference
- Joe H (2006). Generating random correlation matrices based on partial
  correlations. J. Mult. Anal. Vol. 97, 2177--2189.
"""
function rcor(d::Integer, α::Real=1.0)
    if d == 1
        return ones(1, 1)
    elseif d == 2
        ρ = rand(Uniform(-1.0, 1.0))
        return Array([1 ρ; ρ 1])
    else

        function rjm(A, a)
            b     = size(A, 1)
            idx   = 2 ≤ b-1 ? range(2, b-1, step=1) : range(2, b-1, step=-1)
            ρ₁    = A[idx, 1]
            ρ₃    = A[idx, b]
            R₂    = A[idx, idx]
            Rᵢ    = pinv(R₂)
            rcond = 2 * rand(Beta(a, a)) - 1
            t13   = ρ₁' * Rᵢ * ρ₃
            t11   = ρ₁' * Rᵢ * ρ₁
            t33   = ρ₃' * Rᵢ * ρ₃
            t13[1] + rcond * √((1 - t11[1]) * (1 - t33[1]))
        end

        R = Array{Float64}(I, d, d)

        for i=1:d-1
            α₀ = α + (d-2) / 2
            R[i, i+1] = 2 * rand(Beta(α₀, α₀)) - 1
            R[i+1, i] = R[i, i+1]
        end

        for m=2:d-1, j=1:d-m
            r_sub = R[j:j+m, j:j+m]
            α₀ = α + (d - m - 1) / 2
            R[j, j+m] = rjm(r_sub, α₀)
            R[j+m, j] = R[j, j+m]
        end

        return R
    end
end


function setdiag!(A::AbstractMatrix, x::Real)
    @inbounds A[diagind(A)] .= x
end


"""
    z2x(d::UnivariateDistribution, x::AbstractArray)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
function z2x(d::UnivariateDistribution, x::AbstractArray)
    quantile.(d, cdf.(Normal(0,1), x))
end

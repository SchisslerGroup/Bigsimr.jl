struct CorPair
    from::String
    to::String
end

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

    cor_pair = CorPair(from, to)
    @match cor_pair begin
        CorPair("pearson", "spearman") => (6 / π) .* asin.(ρ ./ 2)
        CorPair("pearson", "kendall")  => (2 / π) .* asin.(ρ)
        CorPair("spearman", "pearson") => 2 * sin.(ρ .* π / 6)
        CorPair("spearman", "kendall") => (2 / π) .* asin.(2 * sin.(ρ .* π ./ 6))
        CorPair("kendall", "pearson")  => sin.(ρ .* π / 2)
        CorPair("kendall", "spearman") => (6 / π) .* asin.(sin.(ρ .* π ./ 2) ./ 2)
        CorPair(f, t) => throw(ArgumentError("No matching conversion from '$f' to '$t'. 'from' and 'to' must be any combination of 'pearson', 'spearman', or 'kendall'"))
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
    rcor(d::Integer, k::Integer=1)

Generate a random correlation matrix of size ``d×d``. the parameter `k` is used
to set the factor loadings for ``W``. The code has been adapted from user *amoeba*
from [StackExchange](https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor)
"""
function rcor(d::Integer, k::Integer=1)
    W = randn(Float64, d, k)
    S = W * W' + diagm(rand(Float64, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    S2 .= S2 * S * S2
    setdiag!(S2, 1.0)
    S2
end


function setdiag!(A::AbstractMatrix, x::Real)
    @inbounds A[diagind(A)] .= x
end


"""
    z2x(d::Distribution, x::AbstractArray)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
function z2x(d::Distribution, x::AbstractArray)
    quantile.(d, cdf.(Normal(0,1), x))
end

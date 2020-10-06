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
function hermite(x, n::Int; probabilists::Bool=true)
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


# Type promotion for setdiag()
function promote(A::Matrix{T}, x::S) where {T<:Real, S<:Real}
    TS = promote_type(T, S)
    (Matrix{TS}(A), TS(x))
end

"""
    setdiag(A::Matrix{T}, x::S) where {T<:Real, S<:Real}

Set the diagonal elements of a Matrix to a value. Return the new matrix.
"""
function setdiag(A::Matrix{T}, x::S) where {T<:Real, S<:Real}
    A, x = promote(A, x)
    @inbounds A[diagind(A)] .= x
    A
end

function setdiag!(A::Matrix{Float64}, x::Float64)
    @inbounds A[diagind(A)] .= x
    nothing
end


"""
    z2x(d::UnivariateDistribution, x::AbstractArray)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
function z2x(d::UnivariateDistribution, x::AbstractArray)
    quantile.(d, cdf.(Normal(0,1), x))
end

@memoize function _h(x::Real, n::Int)
    if n == 0
        return length(x) > 1 ? ones(length(x)) : 1
    elseif n == 1
        return x
    else
        return x .* _h(x, n-1) .- (n-1) .* _h(x, n-2)
    end
end

"""
    hermite(x::Real, n::Int, probabilists::Bool=true)

Compute the Hermite polynomials of degree `n` at `x`.

Computes the Probabilists' version by default. The two definitions of the 
Hermite polynomials are each a rescaling of the other. Let ``Heₙ(x)`` denote 
the Probabilists' version, and ``Hₙ(x)`` the Physicists'. Then

```math
H_{n}(x) = 2^{\\frac{n}{2}} He_{n}\\left(\\sqrt{2} x\\right)
```

```math
He_{n}(x) = 2^{-\\frac{n}{2}} H_{n}\\left(\\frac{x}{\\sqrt{2}}\\right)
```
"""
function hermite(x::Real, n::Int, probabilists::Bool=true)
    return probabilists ? _h(x, n) : 2^(n/2) * _h(x*√2, n)
end
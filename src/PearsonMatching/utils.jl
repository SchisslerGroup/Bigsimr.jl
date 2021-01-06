"""
    get_coefs(margin::UnivariateDistribution, n::Int)

Get the ``n^{th}`` degree Hermite Polynomial expansion coefficients for
``F^{-1}[Φ(⋅)]`` where ``F^{-1}`` is the inverse CDF of a probability
distribution and Φ(⋅) is the CDF of a standard normal distribution.

# Notes
The paper describes using Guass-Hermite quadrature using the Probabilists'
version of the Hermite polynomials, while the package `FastGaussQuadrature.jl`
uses the Physicists' version. Because of this, we need to do a rescaling of the
input and the output:

```math
\\frac{1}{k!}\\sum_{s=1}^{m}w_s H_k (t_s) F_{i}^{-1}\\left[\\Phi(t_s)\\right] ⟹
\\frac{1}{\\sqrt{\\pi} \\cdot k!}\\sum_{s=1}^{m}w_s H_k (t_s\\sqrt{2}) F_{i}^{-1}\\left[\\Phi(t_s)\\right]
```
"""
function get_coefs(margin::UD, n::Int)
    c = zeros(Float64, n + 1)
    m = n + 4
    t, w = gausshermite(m)
    @threads for k ∈ 0:n
        # need to do a change of variable
        X = normal_to_margin(margin, t * √2)
        @inbounds c[k+1] = (1 / √π) * sum(w .* hermite.(t * √2, k) .* X) / factorial(k)
    end
    return c
end


"""
    Hϕ(x::Real, n::Int)

We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
"""
function Hϕ(x::Real, n::Int)
    return isinf(x) ? zero(x) : hermite(x, n) * pdf(Normal(), x)
end


"""
    Gn0d(n::Int, A, B, α, β, σAσB_inv)

Calculate the ``n^{th}`` derivative of `G` at `0` where ``ρ_x = G(ρ_z)``

We are essentially calculating a double integral over a rectangular region

```math
\\int_{α_{r-1}}^{α_r} \\int_{β_{s-1}}^{β_s} Φ(z_i, z_j, ρ_z) dz_i dz_j
```

```
(α[r], β[s+1]) +----------+ (α[r+1], β[s+1])
               |          |
               |          |
               |          |
  (α[r], β[s]) +----------+ (α[r+1], β[s])
```
"""
function Gn0d(n::Int, A, B, α, β, σAσB_inv)
    if n == 0
        return 0
    end
    M = length(A)
    N = length(B)
    accu = 0.0
    for r=1:M, s=1:N
        r11 = Hϕ(α[r+1], n-1) * Hϕ(β[s+1], n-1)
        r00 = Hϕ(α[r],   n-1) * Hϕ(β[s],   n-1)
        r01 = Hϕ(α[r],   n-1) * Hϕ(β[s+1], n-1)
        r10 = Hϕ(α[r+1], n-1) * Hϕ(β[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end
    return accu * σAσB_inv
end


"""
    Gn0m(n::Int, A, α, dB, σAσB_inv)

Calculate the ``n^{th}`` derivative of `G` at `0` where ``ρ_x = G(ρ_z)``
"""
function Gn0m(n::Int, A, α, dB, σAσB_inv)
    if n == 0
        return 0
    end
    M = length(A)
    accu = 0.0
    for r=1:M
        accu += A[r] * (Hϕ(α[r+1], n-1) - Hϕ(α[r], n-1))
    end
    m = n + 4
    t, w = gausshermite(m)
    X = MvSim.normal_to_margin(dB, t * √2)
    S = (1 / √π) * sum(w .* hermite.(t * √2, n) .* X)
    return -σAσB_inv * accu * S
end


"""
    solve_poly_pm_one(coef)

Solve a polynomial equation on the interval [-1, 1].
"""
function solve_poly_pm_one(coef)
    P = Polynomial(coef)
	dP = derivative(P)
    r = roots(x->P(x), x->dP(x), -1..1, Krawczyk, 1e-3)
        
    length(r) == 1 && return mid(r[1].interval)
    length(r) == 0 && return NaN
    length(r) > 1 && error("More than one root found in the interval -1..1")
end

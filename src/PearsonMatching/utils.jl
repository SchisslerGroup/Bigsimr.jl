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
    aₖ = zeros(Float64, n + 1)
    m = 2n
    tₛ, wₛ = gausshermite(m)
    tₛ    .= tₛ * sqrt2
    Xₛ = normal_to_margin(margin, tₛ)

    aₖ = [sum(wₛ .* _h.(tₛ, k) .* Xₛ) for k in 0:n]
    
    return invsqrtpi * aₖ ./ factorial.(0:n)
end


"""
    Hp(x::Float64, n::Int)

We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
"""
function Hp(x::Float64, n::Int)
    isinf(x) ? zero(x) : _h(x, n) * _normpdf(x)
end


"""
    Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)

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
function Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)
    if n == 0
        return 0.0
    end
    M = length(A)
    N = length(B)
    accu = 0.0
    for r=1:M, s=1:N
        r11 = Hp(α[r+1], n-1) * Hp(β[s+1], n-1)
        r00 = Hp(α[r],   n-1) * Hp(β[s],   n-1)
        r01 = Hp(α[r],   n-1) * Hp(β[s+1], n-1)
        r10 = Hp(α[r+1], n-1) * Hp(β[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end
    accu * σAσB_inv
end


"""
    Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UnivariateDistribution, σAσB_inv::Float64)

Calculate the ``n^{th}`` derivative of `G` at `0` where ``ρ_x = G(ρ_z)``
"""
function Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UD, σAσB_inv::Float64)

    if n == 0
        return 0.0
    end
    M = length(A)
    accu = 0.0
    for r=1:M
        accu += A[r] * (Hp(α[r+1], n-1) - Hp(α[r], n-1))
    end
    m = n + 4
    t, w = gausshermite(m)
    t .= t * sqrt2
    X = MvSim.normal_to_margin(dB, t)
    S = invsqrtpi * sum(w .* hermite.(t, n) .* X)
    return -σAσB_inv * accu * S
end


"""
    solve_poly_pm_one(coef::Vector{Float64})

Solve a polynomial equation on the interval [-1, 1].
"""
function solve_poly_pm_one(coef::Vector{Float64})
    P = Polynomial(coef)
	dP = derivative(P)
    r = roots(x->P(x), x->dP(x), -1..1, Krawczyk, 1e-3)
        
    length(r) == 1 && return mid(r[1].interval)
    length(r) == 0 && return NaN
    length(r) > 1 && error("More than one root found in the interval -1..1")
end



"""
    hermite(x::Float64, n::Int, probabilists::Bool=true)

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
function hermite(x::Float64, n::Int, probabilists::Bool=true)
    return probabilists ? _h(x, n) : 2^(n/2) * _h(x*sqrt2, n)
end



function _h(x::Float64, n::Int)
    if n == 0
		return 1.0
	elseif n == 1
		return x
	end
	
	Hkp1, Hk, Hkm1 = 0.0, x, 1.0
	for k in 2:n
		Hkp1 = x*Hk - (k-1) * Hkm1
		Hkm1, Hk = Hk, Hkp1
	end
	Hkp1
end
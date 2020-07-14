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
function get_coefs(margin::UnivariateDistribution, n::Int)
    c = Array{Float64,1}(undef, n + 1)
    m = n + 4
    t, w = gausshermite(m)
    for k = 0:1:n
        # need to do a change of variable
        X = z2x(margin, t * √2)
        c[k+1] = (1 / √π) * sum(w .* hermite(t * √2, k) .* X) / factorial(k)
    end
    c
end


# We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
function Hϕ(x::T, n::Int) where T<:Real
    if isinf(x)
        zero(T)
    else
        hermite(x, n) * pdf(Normal(), x)
    end
end


"""
    Gn0d(::Int, A, B, α, β, σAσB_inv)

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
    accu * σAσB_inv
end


"""
    Gn0m(::Int, A, α, dB, σAσB_inv)

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
    X = MvSim.z2x(dB, t * √2)
    S = (1 / √π) * sum(w .* hermite(t * √2, n) .* X)
    -σAσB_inv * accu * S
end


"""
    ρz_bounds(::UnivariateDistribution, ::UnivariateDistribution; ::Int=3)

Compute the lower and upper bounds of possible correlations for a pair of
univariate distributions. The value `n` determines the accuracy of the approximation
of the two distributions.
"""
function ρz_bounds(dA::UnivariateDistribution, dB::UnivariateDistribution; n::Int=7)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)

    # Eq (25)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    # Eq (22)
    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρx_u = c1 * c2 + c2 * sum(kab)

    (clampcor(ρx_l), clampcor(ρx_u))
end


"""
    solvePoly_pmOne(coef)

Solve a polynomial equation on the interval [-1, 1].
"""
function solvePoly_pmOne(coef)
    n = length(coef) - 1
    P(x) = Polynomial(coef)(x)
    dP(x) = Polynomial((1:n) .* coef[2:end])(x)
    r = roots(P, dP, -1..1)
    if length(r) == 0
        NaN
    else
        mid(r[1].interval)
    end
end


# Continuous case -------------------------------------------------------------
"""
function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    μA,
    μB,
    σA,
    σB,
    n::Int=3
)

Estimate the input correlation coefficient `ρz` given the marginal CDFs of two
continuous univariate distributions and the desired correlation coefficient `ρx`.
"""
function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    μA,
    μB,
    σA,
    σB,
    n::Int=3
)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    # Eq (22)
    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b

    # Coefficients for Eq (19)
    coef = c2 .* [a[k+1] * b[k+1] * factorial(k) for k = 1:n]
    coef = [c1 * c2 + c2 * a[1] * b[1] - ρx; coef]

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
        ρx_u = c1 * c2 + c2 * sum(kab)
        clampcor(clamp(ρx, ρx_l, ρx_u))
    else
        r
    end
end


# Shorthand for continuous case
function ρz(ρx, dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int=3)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)
    ρz(ρx, dA, dB, μA, μB, σA, σB, n)
end


# Discrete case ---------------------------------------------------------------
"""
function ρz(
    ρx,
    dA::DiscreteUnivariateDistribution,
    dB::DiscreteUnivariateDistribution,
    σA,
    σB,
    minA,
    minB,
    maxA,
    maxB,
    n::Int=3
)

Estimate the input correlation coefficient `ρz` given the marginal CDFs of two
discrete univariate distributions and the desired correlation coefficient `ρx`.
"""
function ρz(
    ρx,
    dA::DiscreteUnivariateDistribution,
    dB::DiscreteUnivariateDistribution,
    σA,
    σB,
    minA,
    minB,
    maxA,
    maxB,
    n::Int=3
)
    TA = eltype(dA)
    TB = eltype(dB)

    maxA = isinf(maxA) ? TA(quantile(dA, 0.995)) : maxA
    maxB = isinf(maxB) ? TB(quantile(dB, 0.995)) : maxB

    # Support sets
    A = range(minA, maxA, step=1.0)
    B = range(minB, maxB, step=1.0)

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]
    β = [-Inf; quantile.(Normal(), cdf.(dB, B))]

    c2 = 1 / (σA * σB)

    coef = [Gn0d(i, A, B, α, β, c2) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB, n)
        clampcor(clamp(ρx, ρx_l, ρx_u))
    else
        r
    end
end


# Shorthand for discrete case
function ρz(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int=3)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    minB = minimum(dB)
    maxA = maximum(dA)
    maxB = maximum(dB)
    ρz(ρx, dA, dB, σA, σB, minA, minB, maxA, maxB, n)
end


# Mixed case ------------------------------------------------------------------
"""
    function ρz(
        ρx,
        dA::DiscreteUnivariateDistribution,
        dB::ContinuousUnivariateDistribution,
        σA,
        σB,
        minA,
        maxA,
        n::Int=3
    )

Estimate the input correlation coefficient `ρz` given the marginal CDFs of two
mixed support univariate distributions and the desired correlation coefficient `ρx`.
"""
function ρz(
    ρx,
    dA::DiscreteUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    σA,
    σB,
    minA,
    maxA,
    n::Int=3
)
    TA = eltype(dA)
    maxA = isinf(maxA) ? TA(quantile(dA, 0.995)) : maxA
    A = range(minA, maxA, step=1.0)
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]

    c2 = 1 / (σA * σB)

    coef = [Gn0m(i, A, α, dB, c2) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB, n)
        clampcor(clamp(ρx, ρx_l, ρx_u))
    else
        r
    end
end


"""
    function ρz(
        ρx,
        dA::ContinuousUnivariateDistribution,
        dB::DiscreteUnivariateDistribution,
        σA,
        σB,
        minB,
        maxB,
        n::Int=3
    )

Estimate the input correlation coefficient `ρz` given the marginal CDFs of two
mixed support univariate distributions and the desired correlation coefficient `ρx`.
"""
function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::DiscreteUnivariateDistribution,
    σA,
    σB,
    minB,
    maxB,
    n::Int=3
)
    ρz(ρx, dB, dA, σB, σA, minB, maxB, n)
end


# Fallback for mixed case
function ρz(
    ρx,
    dA::DiscreteUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    n::Int=3
)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    maxA = maximum(dA)
    ρz(ρx, dA, dB, σA, σB, minA, maxA, n)
end

function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::DiscreteUnivariateDistribution,
    n::Int=3
)
    σA = std(dA)
    σB = std(dB)
    minB = minimum(dB)
    maxB = maximum(dB)
    ρz(ρx, dB, dA, σB, σA, minB, maxB, n)
end

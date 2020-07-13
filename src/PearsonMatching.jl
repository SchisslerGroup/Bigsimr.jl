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
        c[k+1] = (1 / √π) * sum(w .* He(t * √2, k) .* X) / factorial(k)
    end
    c
end

"""
function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    μA,
    μB,
    σA,
    σB,
    n::Int = 3,
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
    n::Int = 3,
)

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

    # Coefficients for Eq (19)
    coef = c2 .* [a[k+1] * b[k+1] * factorial(k) for k = 1:n]
    coef = [c1 * c2 + c2 * a[1] * b[1] - ρx; coef]

    # Polynomial from Eq (19) and its derivative
    P(x) = Polynomial(coef)(x)
    dP(x) = Polynomial((1:n) .* coef[2:end])(x)

    # Solve the polynomial over the interval [-1, 1]
    r = roots(P, dP, -1..1)

    # If no root is found in the interval [-1, 1], then return the upper or
    # lower bound depending on the sign of ρx. E.g. if ρx = -0.8, then return
    # ρx_l as the root
    if length(r) == 0
        return ρx < 0 ? ρx_l : ρx_u
    else
        return mid(r[1].interval)
    end
end


"""
function ρz(
    ρx,
    dA::ContinuousUnivariateDistribution,
    dB::ContinuousUnivariateDistribution,
    μA,
    μB,
    σA,
    σB,
    n::Int = 3,
)

Estimate the input correlation coefficient `ρz` given the marginal CDFs of two
continuous univariate distributions and the desired correlation coefficient `ρx`.
"""
function ρz(
    ρx,
    dA::DiscreteUnivariateDistribution,
    dB::DiscreteUnivariateDistribution,
    μA,
    μB,
    σA,
    σB,
    n::Int = 3,
)
    minA = minimum(dA)
    maxA = maximum(dA)
    minB = minimum(dB)
    maxB = maximum(dB)

end

function ρz(ρx, dA::UnivariateDistribution, dB::UnivariateDistribution, n::Int=3)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)
    ρz(ρx, dA, dB, μA, μB, σA, σB, n)
end

# Explicit Cases
function ρz(ρx, dA::UnivariateDistribution, dB::UnivariateDistribution, n::Int)
    @mat2 * sin(ρx * π / 6)
end



function ρz_bounds(dA, dB; n::Int = 3)
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

    (ρx_l, ρx_u)
end

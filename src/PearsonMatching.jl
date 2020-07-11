"""
    get_coefs(margin::UnivariateDistribution, n::Integer)

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
function get_coefs(margin::UnivariateDistribution, n::Integer)
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
function rho_z(
    ρx,
    marginᵢ::UnivariateDistribution,
    marginⱼ::UnivariateDistribution,
    μᵢ,
    μⱼ;
    n::Int = 3,
)

Estimate the input correlation coefficient ρ_z given the marginal CDFs of
(Xᵢ, Xⱼ), and the desired correlation coefficient ρ_x
"""
function rho_z(
    ρx,
    marginᵢ::UnivariateDistribution,
    marginⱼ::UnivariateDistribution,
    μᵢ,
    μⱼ,
    σᵢ,
    σⱼ;
    n::Int = 3,
)
    μᵢ = mean(marginᵢ)
    σᵢ = std(marginᵢ)
    μⱼ = mean(marginⱼ)
    σⱼ = std(marginⱼ)

    # Eq (25)
    k = 0:1:n
    a = get_coefs(marginᵢ, n)
    b = get_coefs(marginⱼ, n)

    # Eq (22)
    c1 = -μᵢ * μⱼ
    c2 = 1 / (σᵢ * σⱼ)
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


function rho_z_bounds_cc(marginᵢ, marginⱼ; n::Int = 3)
    μᵢ = mean(marginᵢ)
    σᵢ = std(marginᵢ)
    μⱼ = mean(marginⱼ)
    σⱼ = std(marginⱼ)

    # Eq (25)
    k = 0:1:n
    a = get_coefs(marginᵢ, n)
    b = get_coefs(marginⱼ, n)

    # Eq (22)
    c1 = -μᵢ * μⱼ
    c2 = 1 / (σᵢ * σⱼ)
    kab = factorial.(k) .* a .* b
    ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρx_u = c1 * c2 + c2 * sum(kab)

    (ρx_l, ρx_u)
end

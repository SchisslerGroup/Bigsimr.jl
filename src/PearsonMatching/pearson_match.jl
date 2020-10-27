"""
    ρz(ρx::Real, dA::UD, dB::UD; n::Int=7)

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation given a pair of marginal distributions.
"""
function ρz(ρx::Real, dA::UD, dB::UD; n::Int=7)
    _ρz(ρx, dA, dB, n)
end


function _ρz(ρx::Real, dA::CUD, dB::CUD, n::Int)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)

    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b

    coef = c2 .* [a[k+1] * b[k+1] * factorial(k) for k = 1:n]
    coef = [c1 * c2 + c2 * a[1] * b[1] - ρx; coef]

    r = solve_poly_pm_one(coef)
    if isnan(r)
        ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
        ρx_u = c1 * c2 + c2 * sum(kab)
        return clamp(clamp(ρx, ρx_l, ρx_u), -1, 1)
    else
        return r
    end
end

function _ρz(ρx::Real, dA::DUD, dB::DUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    minB = minimum(dB)
    maxA = maximum(dA)
    maxB = maximum(dB)

    TA = eltype(dA)
    TB = eltype(dB)

    # Support sets
    A = range(minA, maxA, step=1.0)
    B = range(minB, maxB, step=1.0)

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]
    β = [-Inf; quantile.(Normal(), cdf.(dB, B))]

    c2 = 1 / (σA * σB)

    coef = [Gn0d(i, A, B, α, β, c2) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    r = solve_poly_pm_one(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB)
        return clamp(ρx, ρx_l, ρx_u)
    else
        return r
    end
end

function _ρz(ρx::Real, dA::DUD, dB::CUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    maxA = maximum(dA)

    TA = eltype(dA)
    maxA = isinf(maxA) ? TA(quantile(dA, 0.995)) : maxA
    A = range(minA, maxA, step=1.0)
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]

    c2 = 1 / (σA * σB)

    coef = [Gn0m(i, A, α, dB, c2) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    r = solve_poly_pm_one(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB)
        return clamp(ρx, ρx_l, ρx_u)
    else
        return r
    end
end

_ρz(ρx::Real, dA::CUD, dB::DUD, n::Int) = _ρz(ρx, dB, dA, n)

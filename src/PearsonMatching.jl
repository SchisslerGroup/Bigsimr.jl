import Distributions: UnivariateDistribution, 
ContinuousUnivariateDistribution, 
DiscreteUnivariateDistribution


include("PearsonMatching_utils.jl")


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution


"""
    ρz_bounds

Compute the lower and upper bounds of possible correlations for a pair of
univariate distributions. The value `n` determines the accuracy of the 
approximation of the two distributions.
"""
function ρz_bounds end

function ρz_bounds(dA::UD, dB::UD, μA, μB, σA, σB; n::Integer)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρx_u = c1 * c2 + c2 * sum(kab)

    clamp.((ρx_l, ρx_u), -1, 1)
end

ρz_bounds(dA::UD, dB::UD, μA, μB, σA, σB) = ρz_bounds(dA, dB, μA, μB, σA, σB; n=7)

function ρz_bounds(dA::UD, dB::UD)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)
    ρz_bounds(dA, dB, μA, μB, σA, σB)
end


"""
    ρz(ρx, dA::UD, dB::UD; n::Integer=7)

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation given a pair of marginal distributions.
"""
function ρz(ρx, dA::UD, dB::UD; n::Integer)
    _ρz(ρx, dA, dB, n)
end

ρz(ρx, dA::UD, dB::UD) = ρz(ρx, dA, dB; n=7)

function _ρz(ρx, dA::CUD, dB::CUD, n)
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

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
        ρx_u = c1 * c2 + c2 * sum(kab)
        clamp(clamp(ρx, ρx_l, ρx_u), -1, 1)
    else
        r
    end
end

function _ρz(ρx, dA::DUD, dB::DUD, n)
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

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB)
        clamp(ρx, ρx_l, ρx_u)
    else
        r
    end
end

function _ρz(ρx, dA::DUD, dB::CUD, n)
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

    r = solvePoly_pmOne(coef)
    if isnan(r)
        ρx_l, ρx_u = ρz_bounds(dA, dB)
        clamp(ρx, ρx_l, ρx_u)
    else
        r
    end
end

_ρz(ρx, dA::CUD, dB::DUD, n) = _ρz(ρx, dB, dA, n)

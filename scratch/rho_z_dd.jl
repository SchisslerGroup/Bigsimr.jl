using Distributions
using MvSim
import IntervalRootFinding: roots
import Polynomials: Polynomial
using IntervalArithmetic
using Test

function rho_z_dd(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int=3)
    minA = minimum(dA)
    maxA = maximum(dA)
    minB = minimum(dB)
    maxB = maximum(dB)

    TA = eltype(dA)
    TB = eltype(dB)

    maxA = isinf(maxA) ? TA(quantile(dA, 0.999)) : maxA
    maxB = isinf(maxB) ? TB(quantile(dB, 0.999)) : maxB

    # Support sets
    A = range(minA, maxA, step=1.0)
    B = range(minB, maxB, step=1.0)

    # z = Φ⁻¹[F(A)]
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]
    β = [-Inf; quantile.(Normal(), cdf.(dB, B))]

    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)

    # We need to account for when x is ±∞ otherwise Julia will return NaN for 0 * ∞
    function Hϕ(x::T, n::Int) where T<:Real
        if isinf(x)
            zero(T)
        else
            hermite(x, n) * pdf(Normal(), x)
        end
    end


    # Rectangle, r
    #
    # (1,0) +----------+ (1,1)
    #       |          |
    #       |          |
    #       |          |
    # (0,0) +----------+ (0,1)
    function Gn0(n::Int, A, B, α, β, σAσB_inv)
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

    coef = [Gn0(i, A, B, α, β, c2) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    P(x) = Polynomial(coef)(x)

    # Solve the polynomial over the interval [-1, 1]
    r = roots(P, -1..1)
    if length(r) == 0
        return ρx < 0 ? ρx_l : ρx_u
    else
        return mid(r[1].interval)
    end
end

# Values from Table 3
@test -0.939 ≈ rho_z_dd(-0.9, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005
@test -0.624 ≈ rho_z_dd(-0.6, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005
@test -0.311 ≈ rho_z_dd(-0.3, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005
@test  0.310 ≈ rho_z_dd( 0.3, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005
@test  0.618 ≈ rho_z_dd( 0.6, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005
@test  0.925 ≈ rho_z_dd( 0.9, Binomial(20, 0.2), Binomial(20, 0.2)) atol=0.005

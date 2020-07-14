using MvSim
using Distributions
using Test
using IntervalArithmetic

import IntervalRootFinding: roots
import Polynomials: Polynomial
import FastGaussQuadrature: gausshermite

function rho_z_mixed(ρx, dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int=5)
    minA = minimum(dA)
    maxA = maximum(dA)
    A = range(minA, maxA, step=1.0)
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]

    σA = std(dA)
    σB = std(dB)
    c2 = 1 / (σA * σB)

    function Hϕ(x::T, n::Int) where T<:Real
        if isinf(x)
            zero(T)
        else
            hermite(x, n) * pdf(Normal(), x)
        end
    end

    function Gn0_mixed(n::Int, A, α, σAσB_inv, dB)
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

    coef = [Gn0_mixed(i, A, α, c2, dB) / factorial(i) for i=1:n]
    coef = [-ρx; coef]

    P(x) = Polynomial(coef)(x)

    # Solve the polynomial over the interval [-1, 1]
    r = roots(P, -1..1)
    if length(r) == 0
        ρx_l, ρx_u =  ρz_bounds(dA, dB n)
        return clampcor(clamp(ρx, ρx_l, ρx_u))
    else
        return mid(r[1].interval)
    end
end

rho_z_mixed(ρx, dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int=5) = rho_z_mixed(ρx, dB, dA, n)


dA = Binomial(2, 0.2)
dB = Beta(2, 3)

rho_z_mixed(-0.7, dA, dB)

# Values from Table _
@test -0.890 ≈ rho_z_mixed(-0.7, dA, dB) atol=0.005
@test -0.632 ≈ rho_z_mixed(-0.5, dA, dB) atol=0.005
@test -0.377 ≈ rho_z_mixed(-0.3, dA, dB) atol=0.005
@test  0.366 ≈ rho_z_mixed( 0.3, dA, dB) atol=0.005
@test  0.603 ≈ rho_z_mixed( 0.5, dA, dB) atol=0.005
@test  0.945 ≈ rho_z_mixed( 0.8, dA, dB) atol=0.005

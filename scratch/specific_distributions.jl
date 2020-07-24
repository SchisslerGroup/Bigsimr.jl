
using Distributions
using Match

function ρz(dA::UnivariateDistribution, dB::UnivariateDistribution)
    println("Generic function. Should not ever be called")
end

function ρz(dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution)
    println("Both continuous")
end

function ρz(dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution)
    println("Both discrete")
end

function ρz(dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution)
    println("Mixed support")
end

ρz(dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution) = ρz(dB, dA)


ρz(Beta(2, 3), Normal(12, 4.5))
ρz(Beta(2, 3), Binomial(12, .5))
ρz(Binomial(12, .5), Normal(12, 4.5))
ρz(NegativeBinomial(5, 0.3), Binomial(12, .5))
ρz(Uniform(2, 3), Uniform(0, 1))
ρz(Uniform(0, 1), Uniform(0, 1))

function ρz_test(ρx, dA::UnivariateDistribution, dB::UnivariateDistribution)
    ρz_val = @match (dA, dB) begin
        (Uniform(0, 1), Uniform(0, 1)) => 2 * sin(ρx * π / 6)

        (Uniform(0, 1), Binomial(1, 0.5)) || (Binomial(1, 0.5), Uniform(0, 1)) => √(2) * sin(ρx * π / (2*√(3)))

        (Uniform(0, 1), Normal(0, 1)) || (Normal(0, 1), Uniform(0, 1)) => ρx * √(π / 3)

        (Binomial(1, 0.5), Binomial(1, 0.5)) => sin(ρx * π / 2)

        (Binomial(1, 0.5), Normal(0, 1)) || (Normal(0, 1), Binomial(1, 0.5)) => ρx * √(π / 2)

        (Normal(0, 1), LogNormal(0, σ)) => ρx * √(exp(σ^2)-1) / σ
        (LogNormal(0, σ), Normal(0, 1)) => ρx * √(exp(σ^2)-1) / σ

        (LogNormal(0, σ₁), LogNormal(0, σ₂)) => (ρx * √((exp(σ₁^2) - 1)*(exp(σ₂^2) - 1)) + 1) / exp(σ₁ * σ₂)

        (Uniform(0, 1), LogNormal(0, σ)) => (√(2) / σ) * quantile(Normal(0, 1), ρx * √(exp(σ^2) - 1) / (2*√(3)) + 0.5)
        (LogNormal(0, σ), Uniform(0, 1)) => (√(2) / σ) * quantile(Normal(0, 1), ρx * √(exp(σ^2) - 1) / (2*√(3)) + 0.5)

        (Binomial(1, 0.5), LogNormal(0, σ)) => quantile(Normal(0, 1), 0.5 * ρx * √(exp(σ^2) - 1) + 0.5) / σ
        (LogNormal(0, σ), Binomial(1, 0.5)) => quantile(Normal(0, 1), 0.5 * ρx * √(exp(σ^2) - 1) + 0.5) / σ

        (A, B) => ρz(A, B)
    end

    if isnothing(ρz_val)
        # Do more stuff
        println("More complex algorithm required.")
    else
        ρz_val
    end
end


ρz_test(0.4, Uniform(0, 1), Uniform(0, 1))

ρz_test(0.4, Uniform(0, 1), Binomial(1, 0.5))
ρz_test(0.4, Binomial(1, 0.5), Uniform(0, 1))

ρz_test(0.4, Normal(0, 1), Uniform(0, 1))
ρz_test(0.4, Uniform(0, 1), Normal(0, 1))

ρz_test(0.4, Binomial(1, 0.5), Binomial(1, 0.5))

ρz_test(0.4, Binomial(1, 0.5), Normal(0, 1))
ρz_test(0.4, Normal(0, 1), Binomial(1, 0.5))

ρz_test(0.4, Normal(0, 1), LogNormal(0, 0.5))
ρz_test(0.4, LogNormal(0, 0.5), Normal(0, 1))

ρz_test(0.4, LogNormal(0, 0.5), LogNormal(0, 0.1))

ρz_test(0.4, Uniform(0, 1), LogNormal(0, 1))
ρz_test(0.4, LogNormal(0, 1), Uniform(0, 1))

ρz_test(0.4, Binomial(1, 0.5), LogNormal(0, 1))
ρz_test(0.4, LogNormal(0, 1), Binomial(1, 0.5))

ρz_test(0.4, Beta(2, 3), Normal(12, 4.5))
ρz_test(0.4, Beta(2, 3), Binomial(12, .5))
ρz_test(0.4, Binomial(12, .5), Normal(12, 4.5))
ρz_test(0.4, NegativeBinomial(5, 0.3), Binomial(12, .5))
ρz_test(0.4, Uniform(2, 3), Uniform(0, 1))
ρz_test(0.4, Uniform(0, 1), Uniform(0, 1))

using Distributions

function ρz(dA::UnivariateDistribution, dB::UnivariateDistribution)
    println("Generic function")
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



function ρz(dA::Uniform, dB::Uniform)
    if dA == Uniform(0, 1) && dB == Uniform(0, 1)
        println("Both U(0, 1)")
    else
        println("Generic uniform")
    end
end

ρz(Beta(2, 3), Normal(12, 4.5))
ρz(Beta(2, 3), Binomial(12, .5))
ρz(Binomial(12, .5), Normal(12, 4.5))
ρz(NegativeBinomial(5, 0.3), Binomial(12, .5))
ρz(Uniform(2, 3), Uniform(0, 1))
ρz(Uniform(0, 1), Uniform(0, 1))

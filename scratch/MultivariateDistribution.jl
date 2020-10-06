
abstract type AbstractMixedMultivariateDistribution <: MultivariateDistribution end

struct MvDistribution{T<:Real} <: AbstractMixedMultivariateDistribution
    R::Matrix{T}
    X::Tuple{Vararg{UnivariateDistribution}}
    C::Correlation
end

m = (Beta(2, 3), Normal(5, 2.2), Binomial(2, 0.2), Binomial(20, 0.2))
r = cor_randPD(4)
MvDistribution(r, m, Pearson)
using MvSim
using Distributions

struct MixedMultivariateDistribution{T<:Real}
    margins::Tuple{Vararg{UnivariateDistribution}}
    rho::Array{T,2}
    cortype::Symbol
    μ::NTuple{N,T} where {N}
    σ::NTuple{N,T} where {N}
end

function MixedMultivariateDistribution(
    D::Tuple{Vararg{UnivariateDistribution}},
    R::Matrix,
    cortype::Symbol,
)
    μ = mean.(D)
    σ = std.(D)

    @assert cortype ∈ keys(CorrelationTypes)

    MixedMultivariateDistribution(D, R, cortype, μ, σ)
end

margins = (Beta(2, 3), Normal(5, 2.2), Binomial(2, 0.2), Binomial(20, 0.2))
r = rcor(4)
MVD = MixedMultivariateDistribution(margins, r, :S)
MVD.cortype
MVD.μ
MVD.σ
MVD.rho

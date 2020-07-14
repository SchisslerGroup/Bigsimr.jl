using Distributions
using MvSim

struct MixedMultivariateDistribution{T<:Real}
    margins::Tuple{Vararg{UnivariateDistribution}}
    rho::AbstractArray{T,2}
    cortype::String
    μ::NTuple{N,T} where {N}
    σ::NTuple{N,T} where {N}
end

function MixedMultivariateDistribution(
    D::Tuple{Vararg{UnivariateDistribution}},
    R::AbstractArray,
    cortype::String,
)
    μ = mean.(D)
    σ = std.(D)

    cortype = lowercase(cortype)
    @assert cortype ∈ ["pearson", "spearman", "kendall"]

    MixedMultivariateDistribution(D, R, cortype, μ, σ)
end

margins = (Beta(2, 3), Normal(5, 2.2), Binomial(2, 0.2), Binomial(20, 0.2))
r = rcor(4)
MVD = MixedMultivariateDistribution(margins, r, "spearman")
MVD.cortype
MVD.μ
MVD.σ
MVD.rho

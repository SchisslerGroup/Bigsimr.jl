struct MixedMultivariateDistribution{T<:AbstractFloat, S<:Real}
    margins::Tuple{Vararg{UnivariateDistribution}}
    rho::AbstractArray{T,2}
    cortype::String
    μ::NTuple{N,S} where {N}
    σ::NTuple{N,S} where {N}
end

function MixedMultivariateDistribution(
    D::Tuple{Vararg{UnivariateDistribution}},
    R::AbstractArray{T, 2},
    cortype::String,
) where {T <: AbstractFloat}
    μ = mean.(D)
    σ = std.(D)

    cortype = lowercase(cortype)
    @assert cortype ∈ ["pearson", "spearman", "kendall"]

    MixedMultivariateDistribution(D, R, cortype, μ, σ)
end

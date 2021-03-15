using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution, DiscreteUnivariateDistribution
using LinearAlgebra: Cholesky, cholesky

abstract type AbstractCorrelation end
struct Pearson <: AbstractCorrelation end
struct Spearman <: AbstractCorrelation end
struct Kendall <: AbstractCorrelation end

struct CorMat{T<:Real, S<:AbstractMatrix, C<:Union{AbstractCorrelation, Nothing}} <: AbstractMatrix{T}
    mat::S
    chol::Cholesky{T,S}
    type::C
end
CorMat(mat::Matrix) = CorMat(mat, cholesky(mat), nothing)

r = [1.0 0.5; 0.5 1.0]
CorMat(r, cholesky(r), Pearson)
CorMat(r)

struct MvDist{T<:Real, S<:AbstractMatrix, C<:AbstractCorrelation}
    margins::Vector{<:UnivariateDistribution}
    target_cor::CorMat{T, S, C}
    adjust_cor::CorMat{T, S, Pearson}

    function MvDist{T,S,C}(
        m::Vector{<:UnivariateDistribution},
        t::CorMat{T,S,C},
        a::CorMat{T,S,Pearson}
    ) where {T,S,C}
        return new{T,S,C}(m, t, a)
    end
end

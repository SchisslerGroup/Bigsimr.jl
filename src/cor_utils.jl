# Correlation wrappers for package-defined correlation types
cor(x,    ::Pearson)  = cor(x)
cor(x, y, ::Pearson)  = cor(x, y)
cor(x,    ::Spearman) = corspearman(x)
cor(x, y, ::Spearman) = corspearman(x, y)
cor(x,    ::Kendall)  = corkendall(x)
cor(x, y, ::Kendall)  = corkendall(x, y)
cor(x::AbstractVector, ::Correlation) = cor(x)

"""
    cor_convert(ρ::Real, from::Correlation, to::Correlation)

Convert from one type of correlation matrix to another. The possible correlation
types are Pearson, Spearman, or Kendall.
"""
function cor_convert end
cor_convert(ρ, from::C, to::C) where {C<:Correlation} = ρ
cor_convert(ρ, from::Pearson,  to::Spearman) = (6 / π) * asin(ρ / 2)
cor_convert(ρ, from::Pearson,  to::Kendall)  = (2 / π) * asin(ρ)
cor_convert(ρ, from::Spearman, to::Pearson)  = 2 * sin(ρ * π / 6)
cor_convert(ρ, from::Spearman, to::Kendall)  = (2 / π) * asin(2 * sin(ρ * π / 6))
cor_convert(ρ, from::Kendall,  to::Pearson)  = sin(ρ * π / 2)
cor_convert(ρ, from::Kendall,  to::Spearman) = (6 / π) * asin(sin(ρ * π / 2) / 2)
cor_convert(R::Matrix{AbstractFloat}, from::Correlation, to::Correlation) = cor_convert.(R, from, to)



"""
    cor(x, ::Type{<:Correlation})

Compute the correlation matrix. The possible correlation
    types are Pearson, Spearman, or Kendall.
"""
function cor end

cor(x,    ::Type{Pearson})  = cor(x)
cor(x, y, ::Type{Pearson})  = cor(x, y)
cor(x,    ::Type{Spearman}) = corspearman(x)
cor(x, y, ::Type{Spearman}) = corspearman(x, y)
cor(x,    ::Type{Kendall})  = corkendall(x)
cor(x, y, ::Type{Kendall})  = corkendall(x, y)
# cor(x::AbstractVector, ::Correlation) = cor(x)

"""
    cor_convert(ρ::Real, from::Correlation, to::Correlation)

Convert from one type of correlation matrix to another. The possible correlation
types are Pearson, Spearman, or Kendall.
"""
function cor_convert end
cor_convert(ρ, from::Type{C}, to::Type{C}) where {C<:Correlation} = ρ
cor_convert(ρ, from::Type{Pearson},  to::Type{Spearman}) = @. (6 / π) * asin(ρ / 2)
cor_convert(ρ, from::Type{Pearson},  to::Type{Kendall})  = @. (2 / π) * asin(ρ)
cor_convert(ρ, from::Type{Spearman}, to::Type{Pearson})  = @. 2 * sin(ρ * π / 6)
cor_convert(ρ, from::Type{Spearman}, to::Type{Kendall})  = @. (2 / π) * asin(2 * sin(ρ * π / 6))
cor_convert(ρ, from::Type{Kendall},  to::Type{Pearson})  = @. sin(ρ * π / 2)
cor_convert(ρ, from::Type{Kendall},  to::Type{Spearman}) = @. (6 / π) * asin(sin(ρ * π / 2) / 2)
cor_convert(R::AbstractMatrix, from::Type{Correlation}, to::Type{Correlation}) = cor_convert.(R, from, to)


function cov2cor(C::AbstractMatrix)
    s = sqrt.(1.0 ./ diag(C) )
    corr = transpose(s .* transpose(C) ) .* s
    corr[diagind(corr) ] .= 1.0
    return corr
end
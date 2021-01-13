abstract type Correlation end
"""
    Pearson <: Correlation

Pearson's ``r`` product-moment correlation
"""
struct Pearson <: Correlation end
"""
    Spearman <: Correlation

Spearman's ``ρ`` rank correlation
"""
struct Spearman <: Correlation end
"""
    Kendall <: Correlation

Kendall's ``τ`` rank correlation
"""
struct Kendall <: Correlation end



"Multivariate mixed distribution"
struct MvDistribution
    "The correlation matrix"
    ρ::Matrix{<:Real}
    "The marginal distributions"
    F::Vector{<:UD}
    "The type of correlation matrix"
    C::Type{<:Correlation}
end

"""
    margins(D::MvDistribution)
Return the margins of the multivariate distribution.
"""
margins(D::MvDistribution) = D.F
"""
    cor(D::MvDistribution)
Return the correlation matrix of the multivariate distribution.
"""
cor(D::MvDistribution) = D.ρ
"""
    cortype(D::MvDistribution)
Return the correlation matrix type of the multivariate distribution.
"""
cortype(D::MvDistribution) = D.C
"""
    eltype(D::MvDistribution)
Return the eltype of the correlation matrix of the multivariate distribution.
"""
eltype(D::MvDistribution)  = eltype(D.ρ)
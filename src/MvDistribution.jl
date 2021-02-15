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

function Base.show(io::IO, ::MIME"text/plain", D::MvDistribution)
    num_margins = size(D.ρ, 1)
    v, h = displaysize(io)
    num_margins_displayable = v - 6
    first_n = num_margins_displayable ÷ 2
    last_n = num_margins_displayable - first_n
    println(io, "$(num_margins)-marginal MvDistribution with {$(D.C)} target correlation")
    
    if num_margins_displayable > num_margins
        for f in D.f
            println(io, f)
        end
    else
        for i in 1:first_n
            println(io, " ", D.F[i])
        end
        println(io, " ⋮")
        for i in num_margins-last_n:num_margins-1
            println(io, " ", D.F[i])
        end
        print(io, " ", D.F[end])
    end
end
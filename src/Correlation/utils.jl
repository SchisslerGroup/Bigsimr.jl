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


"""
    cor_convert(ρ::Real, from::Correlation, to::Correlation)

Convert from one type of correlation matrix to another. The possible correlation
types are Pearson, Spearman, or Kendall.
"""
function cor_convert end
cor_convert(ρ::AbstractFloat, from::Type{C}, to::Type{C}) where {C<:Correlation} = ρ
cor_convert(ρ::AbstractFloat, from::Type{Pearson},  to::Type{Spearman}) = (6 / π) * asin(ρ / 2)
cor_convert(ρ::AbstractFloat, from::Type{Pearson},  to::Type{Kendall})  = (2 / π) * asin(ρ)
cor_convert(ρ::AbstractFloat, from::Type{Spearman}, to::Type{Pearson})  = 2 * sin(ρ * π / 6)
cor_convert(ρ::AbstractFloat, from::Type{Spearman}, to::Type{Kendall})  = (2 / π) * asin(2 * sin(ρ * π / 6))
cor_convert(ρ::AbstractFloat, from::Type{Kendall},  to::Type{Pearson})  = sin(ρ * π / 2)
cor_convert(ρ::AbstractFloat, from::Type{Kendall},  to::Type{Spearman}) = (6 / π) * asin(sin(ρ * π / 2) / 2)
cor_convert(R::Matrix{<:AbstractFloat}, from::Type{<:Correlation}, to::Type{<:Correlation}) = cor_convert.(copy(R), from, to)


function cor_constrain(R::Matrix{<:AbstractFloat})
    C = copy(R)
    C .= clampcor.(C)
    C .= Symmetric(C)
    C[diagind(C)] .= one(eltype(C))
    C
end

function cov2cor(C::Matrix{<:AbstractFloat})
    D = Diagonal(C)^(-1/2)
    return cor_constrain(D * C * D)
end


"""
    cor_bounds

Compute the pairwise theoretical lower and upper correlation bounds between
distributions.
"""
function cor_bounds(dA::UD, dB::UD, C::Type{<:Correlation}; n::Int=100_000)
    a = rand(dA, n)
    b = rand(dB, n)

    upper = cor(sort!(a), sort!(b), C)
    lower = cor(a, reverse!(b), C)

    return (lower = lower, upper = upper)
end
cor_bounds(dA::UD, dB::UD; n::Int=100_000) = cor_bounds(dA, dB, Pearson; n=n)

function cor_bounds(D::MvDistribution)
    d = length(D.F)

    lower, upper = similar(cor(D)), similar(cor(D))

    @threads for i in collect(subsets(1:d, Val{2}()))
        l, u = cor_bounds(D.F[i[1]], D.F[i[2]], cortype(D))
        lower[i...] = l
        upper[i...] = u
    end

    lower .= cor_constrain(Matrix{eltype(D)}(Symmetric(lower)))
    upper .= cor_constrain(Matrix{eltype(D)}(Symmetric(upper)))

    (lower = lower, upper = upper)
end
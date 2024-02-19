"""
    cor(x[, y], ::Type{<:Correlation})

Compute the correlation matrix of a given type. 

The possible correlation types are:
  * [`Pearson`](@ref)
  * [`Spearman`](@ref)
  * [`Kendall`](@ref)

# Examples
```jldoctest
julia> x = [-1.62169     0.0158613   0.500375  -0.794381
             2.50689     3.31666    -1.3049     2.16058
             0.495674    0.348621   -0.614451  -0.193579
             2.32149     2.18847    -1.83165    2.08399
            -0.0573697   0.39908     0.270117   0.658458
             0.365239   -0.321493   -1.60223   -0.199998
            -0.55521    -0.898513    0.690267   0.857519
            -0.356979   -1.03724     0.714859  -0.719657
            -3.38438    -1.93058     1.77413   -1.23657
             1.57527     0.836351   -1.13275   -0.277048];

julia> cor(x, Pearson)
4×4 Matrix{Float64}:
  1.0        0.86985   -0.891312   0.767433
  0.86985    1.0       -0.767115   0.817407
 -0.891312  -0.767115   1.0       -0.596762
  0.767433   0.817407  -0.596762   1.0

julia> cor(x, Spearman)
4×4 Matrix{Float64}:
  1.0        0.866667  -0.854545   0.709091
  0.866667   1.0       -0.781818   0.684848
 -0.854545  -0.781818   1.0       -0.612121
  0.709091   0.684848  -0.612121   1.0

julia> cor(x, Kendall)
4×4 Matrix{Float64}:
  1.0        0.733333  -0.688889   0.555556
  0.733333   1.0       -0.688889   0.555556
 -0.688889  -0.688889   1.0       -0.422222
  0.555556   0.555556  -0.422222   1.0
```
"""
function cor end

cor(x,    ::Type{Pearson})  = cor(x)
cor(x, y, ::Type{Pearson})  = cor(x, y)
cor(x,    ::Type{Spearman}) = corspearman(x)
cor(x, y, ::Type{Spearman}) = corspearman(x, y)
cor(x,    ::Type{Kendall})  = corkendall(x)
cor(x, y, ::Type{Kendall})  = corkendall(x, y)



"""
    cor_fast(X::AbstractMatrix{<:Real}, C::Type{<:Correlation}=Pearson)

Calculate the correlation matrix in parallel.
"""
function cor_fast(X::AbstractMatrix{<:Real}, C::Type{<:Correlation}=Pearson)
    d = size(X, 2)
    Y = SharedMatrix{eltype(X)}(d, d)
    @threads for i in collect(subsets(1:d, Val{2}()))
        Y[i...] = cor(view(X, :, i[1]), view(X, :, i[2]), C)
    end
    Y .= Symmetric(Y, :U)
    Y[diagind(Y)] .= one(eltype(X))
    return sdata(Y)
end



"""
    cor_convert(X, from, to)

Convert from one type of correlation matrix to another.

The role of conversion in this package is typically used from either Spearman or 
Kendall to Pearson where the Pearson correlation is used in the generation of
random multivariate normal samples. After converting, the correlation matrix
may not be positive semidefinite, so it is recommended to check using 
`LinearAlgebra.isposdef`, and subsequently calling [`cor_nearPD`](@ref).

See also: [`cor_nearPD`](@ref), [`cor_fastPD`](@ref)

The possible correlation types are:
  * [`Pearson`](@ref)
  * [`Spearman`](@ref)
  * [`Kendall`](@ref)

# Examples
```jldoctest
julia> r = [ 1.0       -0.634114   0.551645   0.548993
            -0.634114   1.0       -0.332105  -0.772114
             0.551645  -0.332105   1.0        0.143949
             0.548993  -0.772114   0.143949   1.0];

julia> cor_convert(r, Pearson, Spearman)
4×4 Matrix{Float64}:
  1.0       -0.616168   0.533701   0.531067
 -0.616168   1.0       -0.318613  -0.756979
  0.533701  -0.318613   1.0        0.13758
  0.531067  -0.756979   0.13758    1.0

julia> cor_convert(r, Spearman, Kendall)
4×4 Matrix{Float64}:
  1.0       -0.452063   0.385867    0.383807
 -0.452063   1.0       -0.224941   -0.576435
  0.385867  -0.224941   1.0         0.0962413
  0.383807  -0.576435   0.0962413   1.0

julia> r == cor_convert(r, Pearson, Pearson)
true
```
"""
function cor_convert end

cor_convert(x::Real, ::Type{C},        ::Type{C}) where {C<:Correlation} = x
cor_convert(x::Real, ::Type{Pearson},  ::Type{Spearman}) = clampcor(asin(x / 2) * 6 / π)
cor_convert(x::Real, ::Type{Pearson},  ::Type{Kendall})  = clampcor(asin(x) * 2 / π)
cor_convert(x::Real, ::Type{Spearman}, ::Type{Pearson})  = clampcor(sin(x * π / 6) * 2)
cor_convert(x::Real, ::Type{Spearman}, ::Type{Kendall})  = clampcor(asin(sin(x * π / 6) * 2) * 2 / π)
cor_convert(x::Real, ::Type{Kendall},  ::Type{Pearson})  = clampcor(sin(x * π / 2))
cor_convert(x::Real, ::Type{Kendall},  ::Type{Spearman}) = clampcor(asin(sin(x * π / 2) / 2) * 6 / π)

function cor_convert(X::AbstractMatrix{<:Real}, from::Type{<:Correlation}, to::Type{<:Correlation})
    return cor_constrain!(cor_convert.(X, Ref(from), Ref(to)))
end



"""
    cor_constrain!(X::AbstractMatrix{<:Real}, uplo=:U)

Same as [`cor_constrain`](@ref), except that the matrix `C` is updated in place
to save memory.

# Examples
```jldoctest
julia> a = [ 0.802271   0.149801  -1.1072     1.13451
             0.869788  -0.824395   0.38965    0.965936
            -1.45353   -1.29282    0.417233  -0.362526
             0.638291  -0.682503   1.12092   -1.27018];

julia> cor_constrain!(a)


julia> a
4×4 Matrix{Float64}:
  1.0       0.149801  -1.0        1.0
  0.149801  1.0        0.38965    0.965936
 -1.0       0.38965    1.0       -0.362526
  1.0       0.965936  -0.362526   1.0
```
"""
function cor_constrain!(X::AbstractMatrix{<:Real}, uplo=:U)
    X .= clampcor.(X)
    X .= Symmetric(X, uplo)
    X[diagind(X)] .= one(eltype(X))
    return X
end



"""
    cor_constrain(X::AbstractMatrix{<:Real}, uplo=:U)

Constrain a matrix so that its diagonal elements are 1, off-diagonal elements
are bounded between -1 and 1, and a symmetric view of the upper (if `uplo = :U`)
or lower (if `uplo = :L`) triangle is made.

See also: [`cor_constrain!`](@ref)

# Examples
```jldoctest
julia> a = [ 0.802271   0.149801  -1.1072     1.13451
             0.869788  -0.824395   0.38965    0.965936
            -1.45353   -1.29282    0.417233  -0.362526
             0.638291  -0.682503   1.12092   -1.27018];

julia> cor_constrain(a)
4×4 Matrix{Float64}:
  1.0       0.149801  -1.0        1.0
  0.149801  1.0        0.38965    0.965936
 -1.0       0.38965    1.0       -0.362526
  1.0       0.965936  -0.362526   1.0

julia> cor_constrain(a, :L)
4×4 Matrix{Float64}:
  1.0        0.869788  -1.0   0.638291
  0.869788   1.0       -1.0  -0.682503
 -1.0       -1.0        1.0   1.0
  0.638291  -0.682503   1.0   1.0
```
"""
cor_constrain(X::AbstractMatrix{<:Real}, uplo=:U) = cor_constrain!(copy(X), uplo)



"""
    cov2cor(X::Matrix{<:AbstractFloat})

Transform a covariance matrix into a correlation matrix.

# Details

If ``X \\in \\mathbb{R}^{n \\times n}`` is a covariance matrix, then

```math
\\tilde{X} = D^{-1/2} X  D^{-1/2}, \\quad D = \\mathrm{diag(X)}
```

is the associated correlation matrix.
"""
function cov2cor(X::AbstractMatrix{<:Real})
    D = sqrt(inv(Diagonal(X)))
    return cor_constrain!(D * X * D)
end



"""
    cov2cor!(X::AbstractMatrix{<:Real})

Same as [`cov2cor`](@ref), except that the matrix `C` is updated in place
to save memory.
"""
function cov2cor!(X::AbstractMatrix{<:Real})
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    return cor_constrain!(X)
end

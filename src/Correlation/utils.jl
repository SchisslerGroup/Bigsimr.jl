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
4×4 Array{Float64,2}:
  1.0        0.86985   -0.891312   0.767433
  0.86985    1.0       -0.767115   0.817407
 -0.891312  -0.767115   1.0       -0.596762
  0.767433   0.817407  -0.596762   1.0

julia> cor(x, Spearman)
4×4 Array{Float64,2}:
  1.0        0.866667  -0.854545   0.709091
  0.866667   1.0       -0.781818   0.684848
 -0.854545  -0.781818   1.0       -0.612121
  0.709091   0.684848  -0.612121   1.0

julia> cor(x, Kendall)
4×4 Array{Float64,2}:
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
    cor_convert(R::Matrix{<:AbstractFloat}, from::Type{<:Correlation}, to::Type{<:Correlation})

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
4×4 Array{Float64,2}:
  1.0       -0.616168   0.533701   0.531067
 -0.616168   1.0       -0.318613  -0.756979
  0.533701  -0.318613   1.0        0.13758
  0.531067  -0.756979   0.13758    1.0

julia> cor_convert(r, Spearman, Kendall)
4×4 Array{Float64,2}:
  1.0       -0.452063   0.385867    0.383807
 -0.452063   1.0       -0.224941   -0.576435
  0.385867  -0.224941   1.0         0.0962413
  0.383807  -0.576435   0.0962413   1.0

julia> r == cor_convert(r, Pearson, Pearson)
true
```
"""
cor_convert(R::Matrix{<:AbstractFloat}, from::Type{<:Correlation}, to::Type{<:Correlation}) = cor_convert.(copy(R), from, to)
cor_convert(ρ::AbstractFloat, from::Type{C}, to::Type{C}) where {C<:Correlation} = ρ
cor_convert(ρ::AbstractFloat, from::Type{Pearson},  to::Type{Spearman}) = (6 / π) * asin(ρ / 2)
cor_convert(ρ::AbstractFloat, from::Type{Pearson},  to::Type{Kendall})  = (2 / π) * asin(ρ)
cor_convert(ρ::AbstractFloat, from::Type{Spearman}, to::Type{Pearson})  = 2 * sin(ρ * π / 6)
cor_convert(ρ::AbstractFloat, from::Type{Spearman}, to::Type{Kendall})  = (2 / π) * asin(2 * sin(ρ * π / 6))
cor_convert(ρ::AbstractFloat, from::Type{Kendall},  to::Type{Pearson})  = sin(ρ * π / 2)
cor_convert(ρ::AbstractFloat, from::Type{Kendall},  to::Type{Spearman}) = (6 / π) * asin(sin(ρ * π / 2) / 2)



"""
    cor_constrain!(C::Matrix{<:AbstractFloat}[, uplo=:U])

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
4×4 Array{Float64,2}:
  1.0       0.149801  -1.0        1.0
  0.149801  1.0        0.38965    0.965936
 -1.0       0.38965    1.0       -0.362526
  1.0       0.965936  -0.362526   1.0
```
"""
function cor_constrain!(C::Matrix{<:AbstractFloat}, uplo=:U)
    C .= clampcor.(C)
    C .= Symmetric(C, uplo)
    C[diagind(C)] .= one(eltype(C))
    nothing
end



"""
    cor_constrain(C::Matrix{<:AbstractFloat}[, uplo=:U])

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
4×4 Array{Float64,2}:
  1.0       0.149801  -1.0        1.0
  0.149801  1.0        0.38965    0.965936
 -1.0       0.38965    1.0       -0.362526
  1.0       0.965936  -0.362526   1.0

julia> cor_constrain(a, :L)
4×4 Array{Float64,2}:
  1.0        0.869788  -1.0   0.638291
  0.869788   1.0       -1.0  -0.682503
 -1.0       -1.0        1.0   1.0
  0.638291  -0.682503   1.0   1.0
```
"""
cor_constrain(C::Matrix{<:AbstractFloat}, uplo=:U) = begin R = copy(C); cor_constrain!(R, uplo); R end



"""
    cov2cor(C::Matrix{<:AbstractFloat})

Transform a covariance matrix into a correlation matrix.

# Details

If ``X \\in \\mathbb{R}^{n \\times n}`` is a covariance matrix, then

```math
\\tilde{X} = D^{-1/2} X  D^{-1/2}, \\quad D = \\mathrm{diag(X)}
```

is the associated correlation matrix.
"""
function cov2cor(C::Matrix{<:AbstractFloat})
    D = sqrt(inv(Diagonal(C)))
    return cor_constrain(D * C * D)
end



"""
    cov2cor!(C::Matrix{<:AbstractFloat})

Same as [`cov2cor`](@ref), except that the matrix `C` is updated in place
to save memory.
"""
function cov2cor!(C::Matrix{<:AbstractFloat})
    D = sqrt(inv(Diagonal(C)))
    C .= D * C * D
    cor_constrain!(C)
end



"""
    cor_bounds(dA::UnivariateDistribution, dB::UnivariateDistribution, C::Type{<:Correlation}=Pearson; n_samples::Int=100_000)

Compute the stochastic lower and upper correlation bounds between two marginal
distributions.

This method relies on sampling from each distribution and then estimating the
specified correlation between the sorted samples. Because the samples are random,
there will be some variation in the answer for each call to `cor_bounds`. Increasing
the number of samples will increase the accuracy of the estimate, but will also
take longer to sort. Therefore ≈100,000 samples (the default) are recommended so
that it runs fast while still returning a good estimate.

The possible correlation types are:
  * [`Pearson`](@ref)
  * [`Spearman`](@ref)
  * [`Kendall`](@ref)

# Examples
```julia-repl
julia> using Distributions

julia> A = Normal(78, 10); B = LogNormal(3, 1);

julia> cor_bounds(A, B)
(lower = -0.7646512417819491, upper = 0.7649206069306482)

julia> cor_bounds(A, B, n_samples=Int(1e9))
(lower = -0.7629776825238167, upper = 0.7629762333824238)

julia> cor_bounds(A, B, n_samples=Int(1e4))
(lower = -0.7507010142250724, upper = 0.7551879647701095)

julia> cor_bounds(A, B, Spearman)
(lower = -1.0, upper = 1.0)
```
"""
function cor_bounds(dA::UD, dB::UD, C::Type{<:Correlation}=Pearson; n_samples::Int=100_000)
    a = rand(dA, n_samples)
    b = rand(dB, n_samples)

    sort!(a)
    sort!(b)
    upper = cor(a, b, C)

    reverse!(b)
    lower = cor(a, b, C)

    return (lower = lower, upper = upper)
end



"""
    cor_bounds(D::MvDistribution; n_samples::Int=100_000)

Compute the pairwise stochastic lower and upper correlation bounds between all
marginal distributions.

# Examples
```julia-repl
julia> using Distributions

julia> margins = [Normal(78, 10), LogNormal(3, 1)];

julia> r = [1.0 0.6; 0.6 1.0]
2×2 Array{Float64,2}:
 1.0  0.6
 0.6  1.0

julia> D = MvDistribution(r, margins, Pearson);

julia> bounds = cor_bounds(D);

julia> bounds.lower
2×2 Array{Float64,2}:
  1.0      -0.76892
 -0.76892   1.0

julia> bounds.upper
2×2 Array{Float64,2}:
 1.0       0.768713
 0.768713  1.0
```
"""
function cor_bounds(D::MvDistribution; n_samples::Int=100_000)
    d = length(D.F)
    lower, upper = similar(cor(D)), similar(cor(D))

    @threads for i in collect(subsets(1:d, Val{2}()))
        l, u = cor_bounds(D.F[i[1]], D.F[i[2]], cortype(D), n_samples=n_samples)
        lower[i...] = l
        upper[i...] = u
    end

    lower .= cor_constrain(Matrix{eltype(D)}(Symmetric(lower)))
    upper .= cor_constrain(Matrix{eltype(D)}(Symmetric(upper)))

    (lower = lower, upper = upper)
end
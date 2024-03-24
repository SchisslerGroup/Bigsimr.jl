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

- [`Pearson`](@ref)
- [`Spearman`](@ref)
- [`Kendall`](@ref)

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
function cor_convert(::Real, ::CorType{T1}, ::CorType{T2}) where {T1,T2}
    return error("`cor_convert` is not implemented for cortypes $T1 and $T2")
end

# This method is required for R compatibility.
function cor_convert(xs::AbstractVector{<:Real}, from::CorType, to::CorType)
    return cor_convert.(xs, Ref(from), Ref(to))
end

"""
    cor_convert(X::AbstractMatrix, from, to)

When the input is a matrix, it is assumed to be a correlation matrix, and the resulting
matrix is also constrained to be a correlation matrix (e.g. unit diagonal and off-digonal
elements constrained between -1 and 1).
"""
function cor_convert(X::AbstractMatrix{<:Real}, from::CorType, to::CorType)
    return cor_constrain!(cor_convert.(X, Ref(from), Ref(to)))
end

#! format: off
cor_convert(x::Real, ::CorType{T},         ::CorType{T}) where T = x
cor_convert(x::Real, ::CorType{:Pearson},  ::CorType{:Spearman}) = clampcor(asin(x / 2) * 6 / π)
cor_convert(x::Real, ::CorType{:Pearson},  ::CorType{:Kendall})  = clampcor(asin(x) * 2 / π)
cor_convert(x::Real, ::CorType{:Spearman}, ::CorType{:Pearson})  = clampcor(sinpi(x / 6) * 2)
cor_convert(x::Real, ::CorType{:Spearman}, ::CorType{:Kendall})  = clampcor(asin(sinpi(x / 6) * 2) * 2 / π)
cor_convert(x::Real, ::CorType{:Kendall},  ::CorType{:Pearson})  = clampcor(sinpi(x / 2))
cor_convert(x::Real, ::CorType{:Kendall},  ::CorType{:Spearman}) = clampcor(asin(sinpi(x / 2) / 2) * 6 / π)
#! format: on

"""
    cor_constrain!(X::AbstractMatrix{<:Real})

Same as [`cor_constrain`](@ref), except that the matrix is updated in place to save memory.
"""
function cor_constrain!(X::AbstractMatrix{<:Real})
    X .= clampcor.(X)
    make_symmetric!(X)
    set_diag1!(X)
    return X
end

cor_constrain!(X::Symmetric) = cor_constrain!(X.data)

"""
    cor_constrain(X::AbstractMatrix{<:Real})

Constrain a matrix so that its diagonal elements are 1, off-diagonal elements
are bounded between -1 and 1, and a symmetric view of the upper triangle is made.

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
```
"""
cor_constrain(X::AbstractMatrix{<:Real}) = cor_constrain!(copy(X))

"""
    cov2cor(X::AbstractMatrix{<:Real})

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

Same as [`cov2cor`](@ref), except that the matrix is updated in place to save memory.
"""
function cov2cor!(X::AbstractMatrix{<:Real})
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    return cor_constrain!(X)
end

function cov2cor!(X::Symmetric)
    D = sqrt(inv(Diagonal(X)))
    X.data .= D * X * D
    return cor_constrain!(X)
end

"""
    cor_bounds(d1, d2, cortype, samples)

Compute the stochastic lower and upper correlation bounds between two marginal
distributions.

This method relies on sampling from each distribution and then estimating the
specified correlation between the sorted samples. Because the samples are random,
there will be some variation in the answer for each call to `cor_bounds`. Increasing
the number of samples will increase the accuracy of the estimate, but will also
take longer to sort. Therefore ≈100,000 samples (the default) are recommended so
that it runs fast while still returning a good estimate.

The possible correlation types are:

- [`Pearson`](@ref)
- [`Spearman`](@ref)
- [`Kendall`](@ref)

# Examples

```julia-repl
julia> using Distributions

julia> A = Normal(78, 10); B = LogNormal(3, 1);

julia> cor_bounds(A, B)
(lower = -0.7646512417819491, upper = 0.7649206069306482)

julia> cor_bounds(A, B, 1e6)
(lower = -0.765850375468031, upper = 0.7655170605697716)

julia> cor_bounds(A, B, Pearson)
(lower = -0.7631871539735006, upper = 0.7624398609255689)

julia> cor_bounds(A, B, Spearman, 250_000)
(lower = -1.0, upper = 1.0)
```
"""
function cor_bounds(d1::UD, d2::UD, cortype::CorType, samples::Real)
    n = convert(Int, samples)
    a = rand(d1, n)
    b = rand(d2, n)

    sort!(a)
    sort!(b)

    upper = cor(cortype, a, b)

    reverse!(b)
    lower = cor(cortype, a, b)

    return (lower=lower, upper=upper)
end

cor_bounds(d1::UD, d2::UD) = cor_bounds(d1, d2, Pearson)
cor_bounds(d1::UD, d2::UD, samples::Real) = cor_bounds(d1, d2, Pearson, samples)
cor_bounds(d1::UD, d2::UD, cortype::CorType) = cor_bounds(d1, d2, cortype, 100_000)

"""
    cor_bounds(margins, cortype, samples)

Compute the stochastic pairwise lower and upper correlation bounds between a set
of marginal distributions.

The possible correlation types are:

- [`Pearson`](@ref)
- [`Spearman`](@ref)
- [`Kendall`](@ref)

# Examples

```julia-repl
julia> using Distributions

julia> margins = [Normal(78, 10), Binomial(20, 0.2), LogNormal(3, 1)];

julia> lower, upper = cor_bounds(margins, Pearson);

julia> lower
3×3 Matrix{Float64}:
  1.0       -0.983111  -0.768184
 -0.983111   1.0       -0.702151
 -0.768184  -0.702151   1.0

julia> upper
3×3 Matrix{Float64}:
 1.0       0.982471  0.766727
 0.982471  1.0       0.798379
 0.766727  0.798379  1.0
```
"""
function cor_bounds(margins, cortype::CorType, samples::Real)
    n = convert(Int, samples)

    d = length(margins)
    d > 1 || error("The number of margins must be greater than 1")

    lower = zeros(Float64, d, d)
    upper = zeros(Float64, d, d)

    Base.Threads.@threads for (i, j) in idx_subsets2(d)
        l, u = cor_bounds(margins[i], margins[j], cortype, n)
        lower[i, j] = l
        upper[i, j] = u
    end

    cor_constrain!(lower)
    cor_constrain!(upper)

    return (lower=lower, upper=upper)
end

cor_bounds(margins, cortype::CorType=Pearson) = cor_bounds(margins, cortype, 100_000)
cor_bounds(margins, samples::Real) = cor_bounds(margins, Pearson, samples)

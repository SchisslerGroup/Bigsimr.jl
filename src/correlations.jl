"""
    CorType

A type used for specifiying the type of correlation. Supported correlations are:

    - [`Pearson`](@ref)
    - [`Spearman`](@ref)
    - [`Kendall`](@ref)
"""
struct CorType{T} end

"""
    Pearson

Pearson's ``r`` product-moment correlation
"""
const Pearson = CorType{:Pearson}()

"""
    Spearman

Spearman's ``ρ`` rank correlation
"""
const Spearman = CorType{:Spearman}()

"""
    Kendall

Kendall's ``τ`` rank correlation
"""
const Kendall = CorType{:Kendall}()



"""
    cor(x[, y], ::CorType)

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
Statistics.cor(x, y, cortype::CorType) = _cor(x, y, cortype)
Statistics.cor(X,    cortype::CorType) = _cor(X,    cortype)

_cor(x,    ::CorType{:Pearson})  = Statistics.cor(x)
_cor(x, y, ::CorType{:Pearson})  = Statistics.cor(x, y)
_cor(x,    ::CorType{:Spearman}) = corspearman(x)
_cor(x, y, ::CorType{:Spearman}) = corspearman(x, y)
_cor(x,    ::CorType{:Kendall})  = corkendall(x)
_cor(x, y, ::CorType{:Kendall})  = corkendall(x, y)



"""
    cor_fast(X::AbstractMatrix{<:Real}, C::CorType=Pearson)

Calculate the correlation matrix in parallel.
"""
function cor_fast(X::AbstractMatrix{T}, cortype::CorType=Pearson) where {T<:Real}
    d = size(X, 2)
    Y = SharedMatrix{T}(d, d)

    Base.Threads.@threads for (i, j) in _idx_subsets2(d)
        Y[i,j] = _cor(view(X, :, i), view(X, :, j), cortype)
    end

    _symmetric!(Y)
    _set_diag1!(Y)
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
cor_convert(x::Real, from::CorType, to::CorType) = _cor_convert(x, from, to)

# This method is required for R compatibility.
function cor_convert(xs::AbstractVector{<:Real}, from::CorType, to::CorType)
    return cor_convert.(xs, Ref(from), Ref(to))
end

function cor_convert(X::AbstractMatrix{<:Real}, from::CorType, to::CorType)
    return cor_constrain!(cor_convert.(X, Ref(from), Ref(to)))
end

_cor_convert(x::Real, ::CorType{T}, ::CorType{T}) where T = x
_cor_convert(x::Real, ::CorType{:Pearson},  ::CorType{:Spearman}) = _clampcor(asin(x / 2) * 6 / π)
_cor_convert(x::Real, ::CorType{:Pearson},  ::CorType{:Kendall})  = _clampcor(asin(x) * 2 / π)
_cor_convert(x::Real, ::CorType{:Spearman}, ::CorType{:Pearson})  = _clampcor(sin(x * π / 6) * 2)
_cor_convert(x::Real, ::CorType{:Spearman}, ::CorType{:Kendall})  = _clampcor(asin(sin(x * π / 6) * 2) * 2 / π)
_cor_convert(x::Real, ::CorType{:Kendall},  ::CorType{:Pearson})  = _clampcor(sin(x * π / 2))
_cor_convert(x::Real, ::CorType{:Kendall},  ::CorType{:Spearman}) = _clampcor(asin(sin(x * π / 2) / 2) * 6 / π)




"""
    cor_constrain!(X::AbstractMatrix{<:Real})

Same as [`cor_constrain`](@ref), except that the matrix is updated in place to save memory.
"""
function cor_constrain!(X::AbstractMatrix{<:Real})
    X .= _clampcor.(X)
    _symmetric!(X)
    _set_diag1!(X)
    return X
end



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

Same as [`cov2cor`](@ref), except that the matrix `C` is updated in place to save memory.
"""
function cov2cor!(X::AbstractMatrix{<:Real})
    D = sqrt(inv(Diagonal(X)))
    X .= D * X * D
    return cor_constrain!(X)
end



"""
    cor_randPSD(T, d, k=d-1)

Return a random positive semidefinite correlation matrix where `d` is the
dimension (``d ≥ 1``) and `k` is the number of factor loadings (``1 ≤ k < d``).

See also: [`cor_randPD`](@ref)

# Examples
```julia-repl
julia> cor_randPSD(Float64, 4, 2)
#>4×4 Matrix{Float64}:
 1.0        0.276386   0.572837   0.192875
 0.276386   1.0        0.493806  -0.352386
 0.572837   0.493806   1.0       -0.450259
 0.192875  -0.352386  -0.450259   1.0

julia> cor_randPSD(4, 1)
4×4 Matrix{Float64}:
1.0       -0.800513   0.541379  -0.650587
-0.800513   1.0       -0.656411   0.788824
0.541379  -0.656411   1.0       -0.533473
-0.650587   0.788824  -0.533473   1.0

julia> cor_randPSD(4)
4×4 Matrix{Float64}:
  1.0        0.81691   -0.27188    0.289011
  0.81691    1.0       -0.44968    0.190938
 -0.27188   -0.44968    1.0       -0.102597
  0.289011   0.190938  -0.102597   1.0
```
"""
function cor_randPSD(t::Type{T}, d, k=d-1) where {T<:Real}
    d ≥ 1 || throw(ArgumentError("`d` must be greater than or equal to 1"))
    1 ≤ k < d || throw(ArgumentError("`k` must be greater than '0' and less than `d`"))
    return _cor_randPSD(t, Int(d), Int(k))
end

cor_randPSD(d, k=d-1) = cor_randPSD(Float64, d, k)

function _cor_randPSD(::Type{T}, d::Int, k::Int) where T
    d == 1 && return ones(T, 1, 1)

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    R  = S2 * S * S2

    return cor_constrain!(R)
end



"""
    cor_randPD(T, d, k=d-1)

The same as [`cor_randPSD`](@ref), but calls [`cor_fastPD`](@ref) to ensure that
the returned matrix is positive definite.

# Examples
```julia-repl
julia> cor_randPD(Float64, 4, 2)
#>4×4 Matrix{Float64}:
  1.0        0.458549  -0.33164    0.492572
  0.458549   1.0       -0.280873   0.62544
 -0.33164   -0.280873   1.0       -0.315011
  0.492572   0.62544   -0.315011   1.0

julia> cor_randPD(4, 1)
4×4 Matrix{Float64}:
  1.0        -0.0406469  -0.127517  -0.133308
  -0.0406469   1.0         0.265604   0.277665
  -0.127517    0.265604    1.0        0.871089
  -0.133308    0.277665    0.871089   1.0

julia> cor_randPD(4)
4×4 Matrix{Float64}:
  1.0        0.356488   0.701521  -0.251671
  0.356488   1.0        0.382787  -0.117748
  0.701521   0.382787   1.0       -0.424952
 -0.251671  -0.117748  -0.424952   1.0
```
"""
cor_randPD(t::Type{T}, d, k=d-1) where {T<:Real} = cor_fastPD!(cor_randPSD(t, d, k))
cor_randPD(d, k=d-1) = cor_randPD(Float64, d, k)



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
function cor_bounds(d1::UD, d2::UD, cortype::CorType, samples::Real)
    return _cor_bounds(d1, d2, cortype, Int(samples))
end

cor_bounds(d1::UD, d2::UD) = cor_bounds(d1, d2, Pearson)
cor_bounds(d1::UD, d2::UD, samples::Real) = cor_bounds(d1, d2, Pearson, samples)
cor_bounds(d1::UD, d2::UD, cortype::CorType) = cor_bounds(d1, d2, cortype, 100_000)

function _cor_bounds(d1::UD, d2::UD, cortype::CorType, n::Int)
    a = rand(d1, n)
    b = rand(d2, n)

    sort!(a)
    sort!(b)

    upper = _cor(a, b, cortype)

    reverse!(b)
    lower = _cor(a, b, cortype)

    (lower = lower, upper = upper)
end



"""
    cor_bounds(margins, cortype, samples)

Compute the stochastic pairwise lower and upper correlation bounds between a set
of marginal distributions.

The possible correlation types are:
  * [`Pearson`](@ref)
  * [`Spearman`](@ref)
  * [`Kendall`](@ref)
"""
function cor_bounds(margins::AbstractVector{<:UD}, cortype::CorType, samples::Real)
    return _cor_bounds(margins, cortype, Int(samples))
end

cor_bounds(margins::AbstractVector{<:UD}) = cor_bounds(margins, Pearson)
cor_bounds(margins::AbstractVector{<:UD}, samples::Real) = cor_bounds(margins, Pearson, samples)
cor_bounds(margins::AbstractVector{<:UD}, cortype::CorType) = cor_bounds(margins, cortype, 100_000)

function _cor_bounds(margins::AbstractVector{<:UD}, cortype::CorType, n::Int)
    d = length(margins)
    lower, upper = zeros(Float64, d, d), zeros(Float64, d, d)

    Base.Threads.@threads for (i,j) in _idx_subsets2(d)
        l, u = cor_bounds(margins[i], margins[j], cortype, n)
        lower[i,j] = l
        upper[i,j] = u
    end

    cor_constrain!(lower)
    cor_constrain!(upper)

    (lower = lower, upper = upper)
end



"""
    cor_nearPD(X::AbstractMatrix{<:Real})

Return the nearest positive definite correlation matrix to `X`.

See also: [`cor_nearPSD`](@ref), [`cor_fastPD`](@ref)
"""
cor_nearPD(X) = nearest_cor(X, Newton(;τ=1e-6))

"""
    cor_nearPD!(X::AbstractMatrix{<:Real})

Same as [`cor_nearPD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPSD!`](@ref), [`cor_fastPD!`](@ref)
"""
cor_nearPD!(X) = nearest_cor!(X, Newton(;τ=1e-6))

"""
    cor_nearPSD(X::AbstractMatrix{<:Real})

Return the nearest positive [semi-] definite correlation matrix to `X`.

See also: [`cor_nearPD`](@ref), [`cor_fastPD`](@ref)
"""
cor_nearPSD(X) = nearest_cor(X, Newton(;τ=zero(eltype(X))))

"""
    cor_nearPSD!(X::AbstractMatrix{<:Real})

Same as [`cor_nearPSD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPD!`](@ref), [`cor_fastPD!`](@ref)
"""
cor_nearPSD!(X) = nearest_cor!(X, Newton(;τ=zero(eltype(X))))

"""
    cor_fastPD(X, tau=1e-6)

Return a positive definite correlation matrix that is close to `X`. `τ` is a
tuning parameter that controls the minimum eigenvalue of the resulting matrix.
`τ` can be set to zero if only a positive semidefinite matrix is needed.

See also: [`cor_nearPD`](@ref), [`cor_nearPSD`](@ref)
"""
cor_fastPD(X, tau=1e-6) = nearest_cor(X, DirectProjection(tau))

"""
    cor_fastPD!(X, tau=1e-6)

Same as [`cor_fastPD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPD!`](@ref), [`cor_nearPSD!`](@ref)
"""
cor_fastPD!(X, tau=1e-6) = nearest_cor!(X, DirectProjection(tau))

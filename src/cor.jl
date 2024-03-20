"""
    cor(x[, y], ::CorType)

Compute the correlation matrix of a given type.

The possible correlation types are:

- [`Pearson`](@ref)
- [`Spearman`](@ref)
- [`Kendall`](@ref)

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
function Statistics.cor(::Any, ::Any, ::CorType{T}) where T
    error("Method `cor(x, y, cortype)` is not implemented for cortype $T")
end

function Statistics.cor(::Any, ::CorType{T}) where T
    error("Method `cor(x, cortype)` is not implemented for cortype $T")
end

Statistics.cor(x,    ::CorType{:Pearson})  = cor(x)
Statistics.cor(x, y, ::CorType{:Pearson})  = cor(x, y)
Statistics.cor(x,    ::CorType{:Spearman}) = corspearman(x)
Statistics.cor(x, y, ::CorType{:Spearman}) = corspearman(x, y)
Statistics.cor(x,    ::CorType{:Kendall})  = corkendall(x)
Statistics.cor(x, y, ::CorType{:Kendall})  = corkendall(x, y)



"""
    cor_fast(X::AbstractMatrix{<:Real}, C::CorType=Pearson)

Calculate the correlation matrix in parallel using available threads.
"""
function cor_fast(X::AbstractMatrix{T}, cortype::CorType=Pearson) where T
    d = size(X, 2)
    Y = SharedMatrix{T}(d, d)

    Base.Threads.@threads for (i, j) in _idx_subsets2(d)
        Y[i,j] = cor(view(X, :, i), view(X, :, j), cortype)
    end

    _symmetric!(Y)
    _set_diag1!(Y)
    return sdata(Y)
end

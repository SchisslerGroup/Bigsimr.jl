"""
    cor_randPSD([T::Type{<:AbstractFloat}], d::Int[, k::Int=d-1])

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
function cor_randPSD(T::Type{<:AbstractFloat}, d::Int, k::Int=d-1)
    @assert d ≥ 1
    @assert 1 ≤ k < d

    d == 1 && return ones(T, 1, 1)

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    R = S2 * S * S2

    cor_constrain(R)
end
cor_randPSD(d::Int, k::Int=d-1) = cor_randPSD(Float64, d, k)
cor_randPSD(T::Type{<:AbstractFloat}, d::Real, k::Real=d-1) = cor_randPSD(T, Int(d), Int(k))
cor_randPSD(d::Real, k::Real=d-1) = cor_randPSD(Float64, Int(d), Int(k))


"""
    cor_randPD([T::Type{<:AbstractFloat}], d::Int[, k::Int=d-1])

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
cor_randPD(T::Type{<:AbstractFloat}, d::Int, k::Int=d-1) = cor_fastPD(cor_randPSD(T, d, k))
cor_randPD(d::Int, k::Int=d-1) = cor_randPD(Float64, d, k)
cor_randPD(T::Type{<:AbstractFloat}, d::Real, k::Real=d-1) = cor_randPD(T, Int(d), Int(k))
cor_randPD(d::Real, k::Real=d-1) = cor_randPD(Float64, d, k)

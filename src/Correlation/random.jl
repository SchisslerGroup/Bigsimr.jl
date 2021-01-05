"""
    cor_randPSD([T::Type{<:AbstractFloat}], d::Int[, k::Int=d])

Return a random positive semidefinite correlation matrix.

See also: [`cor_randPD`](@ref)

# Examples
```julia-repl
julia> cor_randPSD(4)
4×4 Array{Float64,2}:
  1.0        0.81691   -0.27188    0.289011
  0.81691    1.0       -0.44968    0.190938
 -0.27188   -0.44968    1.0       -0.102597
  0.289011   0.190938  -0.102597   1.0

julia> cor_randPSD(4, 1)
4×4 Array{Float64,2}:
  1.0       -0.800513   0.541379  -0.650587
 -0.800513   1.0       -0.656411   0.788824
  0.541379  -0.656411   1.0       -0.533473
 -0.650587   0.788824  -0.533473   1.0
```
"""
function cor_randPSD(T::Type{<:AbstractFloat}, d::Int, k::Int=d)
    @assert d ≥ 1
    @assert 1 ≤ k ≤ d

    d == 1 && return ones(T, 1, 1)

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    R = S2 * S * S2

    cor_constrain(R)
end
cor_randPSD(d::Int, k::Int=d) = cor_randPSD(Float64, d, k)



"""
    cor_randPD([T::Type{<:AbstractFloat}], d::Int[, k::Int=d])

The same as [`cor_randPSD`](@ref), but calls [`cor_fastPD`](@ref) to ensure that
the returned matrix is positive definite.

# Examples
```julia-repl
julia> cor_randPSD(4)
4×4 Array{Float64,2}:
  1.0        0.356488   0.701521  -0.251671
  0.356488   1.0        0.382787  -0.117748
  0.701521   0.382787   1.0       -0.424952
 -0.251671  -0.117748  -0.424952   1.0

julia> cor_randPSD(4, 1)
4×4 Array{Float64,2}:
  1.0        -0.0406469  -0.127517  -0.133308
 -0.0406469   1.0         0.265604   0.277665
 -0.127517    0.265604    1.0        0.871089
 -0.133308    0.277665    0.871089   1.0
```
"""
cor_randPD(T::Type{<:AbstractFloat}, d::Int, k::Int=d) = cor_fastPD(cor_randPSD(T, d, k))
cor_randPD(d::Int, k::Int=d) = cor_randPD(Float64, d, k)
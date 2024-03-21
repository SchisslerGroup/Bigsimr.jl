"""
    rmvn(n [, μ], Σ)

Utilizes available threads for fast generation of multivariate normal samples.

# Examples

```julia-repl
julia> μ = [-3, 1, 10];

julia> S = cor_randPD(3)
3×3 Matrix{Float64}:
  1.0        -0.663633  -0.0909108
 -0.663633    1.0        0.151582
 -0.0909108   0.151582   1.0

julia> rmvn(10, μ, S)
10×3 Matrix{Float64}:
 -1.32616   -1.02602     11.0202
 -3.59396    2.84145      8.84367
 -0.441537  -1.53279      8.82931
 -4.69202    2.84618     10.5977
 -2.63359    2.65779      9.8374
 -3.75917    2.07208      8.90139
 -3.00716    0.00897664  10.1173
 -3.00928    0.851214     9.74029
 -3.43021    0.402382     9.51274
 -1.77849    0.157933     9.15944
```
"""
function rmvn(n::Real, μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    d = length(μ)
    r, s = size(Σ)

    (r == s == d)  || throw(DimensionMismatch(
        "The number of margins must match the size of the correlation matrix."))

    n = convert(Int, n)

    return μ' .+ _rmvn_shared(n, Σ)
end

function rmvn(n, Σ::AbstractMatrix{T}) where {T<:Real}
    μ = zeros(T, size(Σ, 2))
    return rmvn(n, μ, Σ)
end


"""
    rvec(n, rho, margins)

Generate samples for a list of marginal distributions and a correaltion structure.

# Examples

```julia-repl
julia> using Distributions

julia> margins = [Normal(3, 1), LogNormal(3, 1), Exponential(3)]

julia> R = [
     1.00 -0.23  0.12
    -0.23  1.00 -0.46
     0.12 -0.46  1.00
];

julia> rvec(10, R, margins)
10×3 Matrix{Float64}:
 3.89423  38.6339    1.30088
 5.87344  11.5582    5.25233
 3.62383  20.4001    3.25627
 3.65075   3.8316    4.48547
 1.62223   9.95032   1.48367
 3.42208  35.0998    0.644814
 1.82689  58.417     0.580125
 4.73678   4.75506  11.2741
 1.92511   9.44913   0.651013
 3.19883  39.3707    0.581781
```
"""
function rvec(n, rho::AbstractMatrix{T}, margins) where {T<:Real}
    d = length(margins)
    r, s = size(rho)

    is_correlation(rho) || throw(ArgumentError("`rho` must be a valid correlation matrix"))
    (r == s == d) || throw(DimensionMismatch(
        "The number of margins must match the size of the correlation matrix."))

    n = convert(Int, n)

    Z = _rmvn_shared(n, rho)

    Base.Threads.@threads for i in 1:d
        @inbounds @view(Z[:,i]) .= _norm2margin(margins[i], @view(Z[:,i]))
    end

    return sdata(Z)
end

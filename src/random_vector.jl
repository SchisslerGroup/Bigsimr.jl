"""
    rmvn(n[, μ], Σ)

Fast parrallel generation of multivariate normal samples.
"""
function rmvn(n, μ, Σ)
    return μ' .+ _rmvn(n, Σ)
end

function rmvn(n, Σ)
    μ = zeros(eltype(Σ), size(Σ, 2))
    return rmvn(n, μ, Σ)
end



"""
    rvec(n, rho, margins)

Generate samples for a list of marginal distributions and a correaltion structure.

# Examples
```julia-repl
julia> using Distributions

julia> import LinearAlgebra: diagind

julia> margins = [Normal(3, 1), LogNormal(3, 1), Exponential(3)]

julia> R = fill(0.5, 3, 3); R[diagind(R)] .= 1.0;

julia> rvec(10, R, margins)
#>10×3 Matrix{Float64}:
 3.71109  82.0696   8.14332
 1.23022  38.8599   2.64595
 2.07222   3.76843  1.13465
 2.82434  11.6953   0.891066
 2.37599  10.0552   1.80555
 2.85431  25.4935   3.40865
 3.85253  21.2241   3.67532
 3.70605  59.2439   2.02502
 2.34066   1.89257  0.619948
 3.83507  16.787    0.66837
```
"""
function rvec(n, rho, margins::AbstractVector{<:UD})
    T = eltype(rho)
    d = length(margins)
    r,s = size(rho)

    (r == s == d) || throw(DimensionMismatch(
        "The number of margins must match the size of the correlation matrix."))
    
    iscorrelation(rho) || throw(ValidCorrelationError())

    Z = SharedMatrix{T}(_rmvn(n, rho))

    @inbounds @threads for i in 1:d
        Z[:,i] = _norm2margin(margins[i], Z[:,i])
    end
    
    return sdata(Z)
end

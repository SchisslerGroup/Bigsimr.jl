"""
    pearson_match(ρ::Float64, dA::UnivariateDistribution, dB::UnivariateDistribution; n::Int=7)

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation given a pair of marginal distributions.

See also: [`pearson_bounds`](@ref)

# Examples
```jldoctest
julia> using Distributions

julia> A = Normal(78, 10); B = LogNormal(3, 1);

julia> pearson_match(0.76, A, B)
0.9962326957682248
```

The target correlation may not be feasible (see [`pearson_bounds`](@ref)), in 
which case the match to the nearest lower or upper bound is returned.

```julia-repl
julia> pearson_match(0.9, A, B)
┌ Warning: The target correlation is not feasible. Returning the match to the nearest bound instead.
[...]
0.9986891675056113
```
"""
function pearson_match(ρ::Float64, dA::UD, dB::UD; n::Int=7, convert::Bool=true)
    
    # Check support set size of any discrete distributions and convert if necessary
    if convert
        cutoff = 200

        if typeof(dA) <: DUD
            maxA = maximum(dA)
            if isinf(maxA) maxA = quantile(dA, 0.99_999) end
            if maxA > cutoff
                @warn "$dA was converted to a GSDist for computational efficiency"
                dA = GSDist(dA)
            end
        end

        if typeof(dB) <: DUD
            maxB = maximum(dB)
            if isinf(maxB) maxB = quantile(dB, 0.99_999) end
            if maxB > cutoff
                @warn "$dB was converted to a GSDist for computational efficiency"
                dB = GSDist(dB)
            end
        end
    end
    
    _pearson_match(ρ, dA, dB, n)
end

# Continuous-Continuous case
function _pearson_match(ρ::Float64, dA::CUD, dB::CUD, n::Int)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)

    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = c2 .* a[k+1] * b[k+1] * factorial(k)
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - ρ

    r = solve_poly_pm_one(coef)
    length(r) == 1 && return r
    !isnan(r) && nearest_root(ρ, r)

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    @warn "The target correlation is not feasible. Returning the match to the nearest bound instead."
    ρ_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρ_u = c1 * c2 + c2 * sum(kab)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end

# Discrete-Discrete case
function _pearson_match(ρ::Float64, dA::DUD, dB::DUD, n::Int)
    maxA = maximum(dA)
    maxB = maximum(dB)
    maxA = isinf(maxA) ? quantile(dA, 0.99_999) : maxA
    maxB = isinf(maxB) ? quantile(dB, 0.99_999) : maxB

    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    minB = minimum(dB)

    # Support sets
    A = minA:maxA
    B = minB:maxB

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    α = [-Inf; _norminvcdf.(cdf.(dA, A))]
    β = [-Inf; _norminvcdf.(cdf.(dB, B))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = Gn0d(k, A, B, α, β, c2) / factorial(k)
    end
    coef[1] = -ρ

    r = solve_poly_pm_one(coef)
    length(r) == 1 && return r
    !isnan(r) && nearest_root(ρ, r)

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    ρ_l, ρ_u = pearson_bounds(dA, dB)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end

# Discrete-Continuous case
function _pearson_match(ρ::Float64, dA::DUD, dB::CUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    maxA = maximum(dA)

    maxA = isinf(maxA) ? quantile(dA, 0.99) : maxA

    A = minA:maxA
    α = [-Inf; _norminvcdf.(cdf.(dA, A))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = Gn0m(k, A, α, dB, c2) / factorial(k)
    end
    coef[1] = -ρ

    r = solve_poly_pm_one(coef)
    length(r) == 1 && return r
    !isnan(r) && nearest_root(ρ, r)

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    ρ_l, ρ_u = pearson_bounds(dA, dB)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end
_pearson_match(ρ::Float64, dA::CUD, dB::DUD, n::Int) = _pearson_match(ρ, dB, dA, n)


"""
    pearson_match(ρ::Matrix{Float64}, margins::Vector{<:UD})

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation matrix given a set of marginal distributions.

See also: [`pearson_bounds`](@ref)
"""
function pearson_match(ρ::Matrix{Float64}, margins::Vector{<:UD})
    !(length(margins) == size(ρ, 1) == size(ρ, 2)) && throw(DimensionMismatch("The number of margins must be the same size as the correlation matrix."))

    d = length(margins)
    R = SharedMatrix{Float64}(d, d)

    # Calculate the pearson matching pairs
    @threads for i in collect(subsets(1:d, Val{2}()))
        @inbounds R[i...] = pearson_match(ρ[i...], margins[i[1]], margins[i[2]])
    end
    R = Matrix{Float64}(Symmetric(sdata(R)))

    # Ensure that the resulting correlation matrix is positive definite
    iscorrelation(R) ? R : cor_fastPD(R)
end

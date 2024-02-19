"""
    pearson_match(p::Real, dA::UnivariateDistribution, dB::UnivariateDistribution; n=7)

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
function pearson_match(p::Real, d1::UD, d2::UD; n=7, convert=true)
    # Check support set size of any discrete distributions and convert if necessary
    if convert
        cutoff = 200

        if typeof(d1) <: DUD
            max1 = maximum(d1)

            if isinf(max1) 
                max1 = quantile(d1, 0.99_999) 
            end
            
            if max1 > cutoff
                @warn "$d1 was converted to a GSDist for computational efficiency"
                d1 = GSDist(d1)
            end
        end

        if typeof(d2) <: DUD
            max2 = maximum(d2)

            if isinf(max2)
                max2 = quantile(d2, 0.99_999)
            end

            if max2 > cutoff
                @warn "$d2 was converted to a GSDist for computational efficiency"
                d2 = GSDist(d2)
            end
        end
    end
    
    return _pearson_match(p, d1, d2, Int(n))
end

# Continuous-Continuous case
function _pearson_match(p::Real, d1::CUD, d2::CUD, n::Int)
    μ1 = mean(d1)
    μ2 = mean(d2)
    s1 = std(d1)
    s2 = std(d2)

    k = 0:1:n
    a = _get_coefs(d1, n)
    b = _get_coefs(d2, n)

    c1 = -μ1 * μ2
    c2 = 1 / (s1 * s2)
    kab = factorial.(k) .* a .* b

    coef = zeros(Float64, n+1)
    for i in 1:n
        coef[i+1] = c2 .* a[i+1] * b[i+1] * factorial(i)
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - p

    r = _solve_poly_pm_one(coef)
    length(r) > 1 && return _nearest_root(p, r)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    @warn "The target correlation is not feasible. Returning the match to the nearest bound instead."
    pl = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    pu = c1 * c2 + c2 * sum(kab)
    return p > 0 ? _pearson_match(pu-0.001, d1, d2, n) : _pearson_match(pl+0.001, d1, d2, n)
end

# Discrete-Discrete case
function _pearson_match(p::Real, d1::DUD, d2::DUD, n::Int)
    max1 = maximum(d1)
    max2 = maximum(d2)
    max1 = isinf(max1) ? quantile(d1, 0.99_999) : max1
    max2 = isinf(max2) ? quantile(d2, 0.99_999) : max2

    s1 = std(d1)
    s2 = std(d2)
    min1 = minimum(d1)
    min2 = minimum(d2)

    # Support sets
    A = min1:max1
    B = min2:max2

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    a = [-Inf; _norminvcdf.(cdf.(d1, A))]
    b = [-Inf; _norminvcdf.(cdf.(d2, B))]

    c2 = 1 / (s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = _Gn0d(k, A, B, a, b, c2) / factorial(k)
    end
    coef[1] = -p

    r = _solve_poly_pm_one(coef)
    length(r) > 1 && return _nearest_root(p, r)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    pl, pu = pearson_bounds(d1, d2)
    return p > 0 ? _pearson_match(pu-0.001, d1, d2, n) : _pearson_match(pl+0.001, d1, d2, n)
end

# Discrete-Continuous case
function _pearson_match(p::Real, d1::DUD, d2::CUD, n::Int)
    s1 = std(d1)
    s2 = std(d2)
    min1 = minimum(d1)
    max1 = maximum(d1)

    max1 = isinf(max1) ? quantile(d1, 0.99) : max1

    A = min1:max1
    a = [-Inf; _norminvcdf.(cdf.(d1, A))]

    c2 = 1 / (s1 * s2)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = _Gn0m(k, A, a, d2, c2) / factorial(k)
    end
    coef[1] = -p

    r = _solve_poly_pm_one(coef)
    length(r) > 1 && return _nearest_root(p, r)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    pl, pu = pearson_bounds(d1, d2)
    return p > 0 ? _pearson_match(pu-0.001, d1, d2, n) : _pearson_match(pl+0.001, d1, d2, n)
end

_pearson_match(p::Real, d1::CUD, d2::DUD, n::Int) = _pearson_match(p, d2, d1, n)


"""
    pearson_match(p::AbstractMatrix{<:Real}, margins::AbstractVector{<:UnivariateDistribution})

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation matrix given a set of marginal distributions.

See also: [`pearson_bounds`](@ref)
"""
function pearson_match(X::AbstractMatrix{<:Real}, margins::AbstractVector{<:UD})
    d = length(margins)
    r, s = size(X)
    (r == s == d) || throw(DimensionMismatch(
        "The number of margins must match the size of the correlation matrix."))

    R = SharedMatrix{Float64}(d, d)

    # Calculate the pearson matching pairs
    @threads for i in collect(subsets(1:d, Val{2}()))
        @inbounds R[i...] = pearson_match(X[i...], margins[i[1]], margins[i[2]])
    end

    R = Symmetric(sdata(R))

    # Ensure that the resulting correlation matrix is positive definite
    return iscorrelation(R) ? R : cor_fastPD!(R)
end

"""
    pearson_bounds(dA::UnivariateDistribution, dB::UnivariateDistribution; n=7)

Compute the theoretical lower and upper Pearson correlation values for a pair of 
univariate distributions.

See also: [`pearson_match`](@ref)

# Examples
```jldoctest
julia> using Distributions

julia> A = Normal(78, 10); B = LogNormal(3, 1);

julia> pearson_bounds(A, B)
(lower = -0.7628739783668452, upper = 0.762873978367046)
```
"""
function pearson_bounds(d1::UD, d2::UD; n=7)
    μ1 = mean(d1)
    s1 = std(d1)
    μ2 = mean(d2)
    s2 = std(d2)

    return _pearson_bounds(d1, d2, μ1, μ2, s1, s2, Int(n))
end



function _pearson_bounds(d1, d2, μ1, μ2, s1, s2, n)
    k = 0:1:n
    a = _get_coefs(d1, n)
    b = _get_coefs(d2, n)

    c1 = -μ1 * μ2
    c2 = 1 / (s1 * s2)
    kab = factorial.(k) .* a .* b
    rl = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ru = c1 * c2 + c2 * sum(kab)

    rl, ru = clampcor.((rl, ru))
    (lower = rl, upper = ru)
end



"""
    pearson_bounds(margins::AbstractVector{<:UnivariateDistribution})

Compute the theoretical lower and upper Pearson correlation values for a set of 
univariate distributions.
    
See also: [`pearson_match`](@ref)

# Examples
```jldoctest
julia> using Distributions

julia> m = [Normal(78, 10), LogNormal(3, 1)];

julia> b = pearson_bounds(m);


julia> b.lower
2×2 Matrix{Float64}:
  1.0       -0.762874
 -0.762874   1.0

julia> b.upper
2×2 Matrix{Float64}:
 1.0       0.762874
 0.762874  1.0
```
"""
function pearson_bounds(margins::AbstractVector{<:UD})
    d = length(margins)
    lower, upper = zeros(Float64, d, d), zeros(Float64, d, d)

    @threads for i in collect(subsets(1:d, Val{2}()))
        l, u = pearson_bounds(margins[i[1]], margins[i[2]])
        lower[i...] = l
        upper[i...] = u
    end

    lower .= Symmetric(lower)
    cor_constrain!(lower)

    upper .= Symmetric(upper)
    cor_constrain!(upper)
    
    (lower = lower, upper = upper)
end

"""
    pearson_bounds(dA::UnivariateDistribution, dB::UnivariateDistribution; n::Int=7)

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
function pearson_bounds(dA::UD, dB::UD; n::Int=7)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)
    _pearson_bounds(dA, dB, μA, μB, σA, σB, n)
end
function _pearson_bounds(dA::UD, dB::UD, μA, μB, σA, σB, n)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρ_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρ_u = c1 * c2 + c2 * sum(kab)

    ρ_l, ρ_u = clampcor.((ρ_l, ρ_u))
    (lower = ρ_l, upper = ρ_u)
end


"""
    pearson_bounds(margins::Vector{<:UnivariateDistribution})

Compute the theoretical lower and upper Pearson correlation values for a set of 
univariate distributions.
    
See also: [`pearson_match`](@ref)

# Examples
```jldoctest
julia> using Distributions

julia> m = [Normal(78, 10), LogNormal(3, 1)];

julia> b = pearson_bounds(m);

julia> b.lower
#>2×2 Array{Float64,2}:
  1.0       -0.762874
 -0.762874   1.0

julia> b.upper
#>2×2 Array{Float64,2}:
 1.0       0.762874
 0.762874  1.0
```
"""
function pearson_bounds(margins::Vector{<:UD})
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

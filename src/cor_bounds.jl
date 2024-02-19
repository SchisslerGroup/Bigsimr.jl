"""
    cor_bounds(d1::UnivariateDistribution, d2::UnivariateDistribution, cortype::CorType=Pearson; n_samples=100_000)

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
function cor_bounds(d1::UD, d2::UD, cortype::CorType=Pearson; n_samples=100_000)
    return _cor_bounds(d1, d2, cortype, Int(n_samples))
end

function _cor_bounds(d1::UD, d2::UD, cortype::CorType, n::Int)
    a = rand(d1, n)
    b = rand(d2, n)

    sort!(a)
    sort!(b)

    upper = cor(a, b, cortype)

    reverse!(b)
    lower = cor(a, b, cortype)

    (lower = lower, upper = upper)
end



"""
    cor_bounds(margins::AbstractVector{<:UnivariateDistribution}, cortype::CorType=Pearson; n_samples=100_000)

Compute the stochastic pairwise lower and upper correlation bounds between a set
of marginal distributions.

The possible correlation types are:
  * [`Pearson`](@ref)
  * [`Spearman`](@ref)
  * [`Kendall`](@ref)
"""
function cor_bounds(margins::AbstractVector{<:UD}, cortype::CorType=Pearson; n_samples=100_000)
    return _cor_bounds(margins, cortype, Int(n_samples))
end

function _cor_bounds(margins::AbstractVector{<:UD}, cortype::CorType, n::Int)
    d = length(margins)
    lower, upper = zeros(Float64, d, d), zeros(Float64, d, d)

    @threads for i in collect(subsets(1:d, Val{2}()))
        l, u = cor_bounds(margins[i[1]], margins[i[2]], cortype; n_samples=n)
        lower[i...] = l
        upper[i...] = u
    end

    cor_constrain!(lower)
    cor_constrain!(upper)

    (lower = lower, upper = upper)
end

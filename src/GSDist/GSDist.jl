"""
    GSDist <: ContinuousUnivariateDistribution

Generalized S-Distribution for approximating univariate distributions.
"""
struct GSDist <: ContinuousUnivariateDistribution
    F₀::Real
    x₀::Real
    α::Real
    g::Real
    k::Real
    γ::Real
    dist::Union{UnivariateDistribution, Nothing} # underlying distribution
    function GSDist(F₀, x₀, α, g, k, γ, dist)
        F₀ < 0 || F₀ > 1 && throw(DomainError(F₀, "F₀ must be between 0 and 1"))
        α ≤ 0 && throw(DomainError(α, "α must be a non-negative real number"))
        g ≤ 0 && throw(DomainError(g, "g must be a non-negative real number"))
        k ≤ 0 && throw(DomainError(k, "k must be a non-negative real number"))
        γ ≤ 0 && throw(DomainError(γ, "γ must be a non-negative real number"))

        new(F₀, x₀, α, g, k, γ, dist)
    end
end

function GSDist(dist::UnivariateDistribution, F₀::Real=0.5;
    n::Int=21, diff::Function=three_point_midpoint, h::Real=1.0)
    F₀ < 0 || F₀ > 1 && throw(DomainError(F₀, "F₀ must be between 0 and 1"))
    F₀ = Float64(F₀)

    l,u = extrema(dist)
    if isinf(l) l = quantile(dist, 1e-6) end
    if isinf(u) u = quantile(dist, 1 - 1e-6) end
    ql, qu = cdf.(dist, (l,u))

    q = range(ql, qu, length=n)
    X = typeof(dist) <: DiscreteUnivariateDistribution ? Set(quantile.(dist, q)) : quantile.(dist, q)

    FX = cdf.(dist, X)
    dF = y->diff(x->cdf(dist,x), y, h)
    fX = typeof(dist) <: DiscreteUnivariateDistribution ? dF.(X) : pdf.(dist, X)

    f = (t, p) -> p[1] * t.^p[2] .* (1.0 .- t.^p[3]).^p[4]
    
    p0 = [1, 0.5, 1, 0.5]
    fit = curve_fit(f, FX, fX, p0)
    p = coef(fit)

    GSDist(F₀, quantile(dist, F₀), p..., dist)
end

function Base.show(io::IO, ::MIME"text/plain", G::GSDist)
    α, g, k, γ = params(G)
    println(io, "Generalized S-Distribution ($(G.dist))")
    println(io, " α: $α")
    println(io, " g: $g")
    println(io, " k: $k")
    print(io, " γ: $γ")
end

params(G::GSDist) = (G.α, G.g, G.k, G.γ)

function quantile(D::GSDist, p::Real)
    q = _gsd_quantile(p, D.F₀, D.x₀, D.α, D.g, D.k, D.γ)
    if isnan(q)
        @debug "Unable to calculate the quantile of the GSDist. Falling back to the quantile of the underlying distribution."
        quantile(D.dist, p)
    else
        q
    end
end

function mean(G::GSDist)
    m = _gsd_mean(G.F₀, G.x₀, G.α, G.g, G.k, G.γ)
    if isnan(m)
        @debug "Unable to calculate the mean of the GSDist. Falling back to the mean of the underlying distribution."
        mean(G.dist)
    else
        m
    end
end

function var(G::GSDist)
    m1 = mean(G)
    m2 = try
        _gsd_moment(G.F₀, G.x₀, G.α, G.g, G.k, G.γ, moment=2)
    catch y
        if isa(y, DomainError)
            @debug "Unable to calculate the variance of the GSDist. Falling back to the variance of the underlying distribution."
            return var(G.dist)
        end
    end
    m2 - m1^2
end

std(G::GSDist) = sqrt(var(G))

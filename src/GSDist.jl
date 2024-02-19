two_point_forward(f, x, h=1) = (f(x+h) - f(x)) / h
three_point_endpoint(f, x, h=1) = (-3f(x) + 4f(x+h)-f(x+2h)) / 2h
three_point_midpoint(f, x, h=1) = (f(x+h) - f(x-h)) / 2h
five_point_endpoint(f, x, h=1) = (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / 12h
five_point_midpoint(f, x, h=1) = (f(x-2h) - 8f(x-h) + 8f(x+h)-f(x+2h)) / 12h



_beta_inc(z1, z2, a, b) = inv(a) * (z2^a * _₂F₁(a,1-b,a+1,z2) - z1^a * _₂F₁(a,1-b,a+1,z1))
_beta_inc(x, a, b) = _beta_inc(0, x, a, b)
_beta_inc(a, b) = _beta_inc(0, prevfloat(1.0), a, b)



_gsd_quantile(p, F₀, x₀, a, g, k, y) = x₀ + _beta_inc(F₀^k, p^k, (1-g)/k, 1-y) / (a*k)



function _gsd_mean(F₀, x₀, a, g, k, y)
	z1 = F₀^k
	a1 = (1 - g) / k
	b1 = 1 - y
	a2 = (2 - g) / k
	b2 = 1 - y
	return x₀ + (_beta_inc(z1, prevfloat(1.0), a1, b1) - _beta_inc(a2, b2)) * inv(a * k)
end



function _gsd_moment(F₀, x₀, a, g, k, y; moment=1)
    z1 = F₀^k
    a1 = (1 - g) / k
    b1 = 1 - y
    c1 = inv(a * k)
    
    f = q -> (x₀ + c1 * _beta_inc(z1, q^k, a1, b1))^moment
    return first(quadgk(f, 0, 1, atol=1e-4)) 
end



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
    dist::Union{UD, Nothing} # underlying distribution

    function GSDist(F₀, x₀, a, g, k, y, dist)
        F₀ < 0 || F₀ > 1 && throw(DomainError(F₀, "F₀ must be between 0 and 1"))
        a ≤ 0 && throw(DomainError(a, "α must be a non-negative real number"))
        g ≤ 0 && throw(DomainError(g, "g must be a non-negative real number"))
        k ≤ 0 && throw(DomainError(k, "k must be a non-negative real number"))
        y ≤ 0 && throw(DomainError(y, "γ must be a non-negative real number"))

        new(F₀, x₀, a, g, k, y, dist)
    end
end

function GSDist(dist::UD, F₀::Real=0.5;
    n::Int=21, diff::Function=three_point_midpoint, h::Real=1.0)

    F₀ < 0 || F₀ > 1 && throw(DomainError(F₀, "F₀ must be between 0 and 1"))
    F₀ = Float64(F₀)

    l, u = extrema(dist)

    if isinf(l)
        l = quantile(dist, 1e-6)
    end

    if isinf(u)
        u = quantile(dist, 1 - 1e-6)
    end

    ql, qu = cdf.(dist, (l,u))

    q = range(ql, qu; length=n)
    X = typeof(dist) <: DUD ? Set(quantile.(dist, q)) : quantile.(dist, q)

    FX = cdf.(dist, X)
    dF = y -> diff(x -> cdf(dist, x), y, h)
    fX = typeof(dist) <: DUD ? dF.(X) : pdf.(dist, X)

    f = (t, p) -> p[1] * t.^p[2] .* (1.0 .- t.^p[3]).^p[4]
    
    p0 = [1, 0.5, 1, 0.5]
    fit = curve_fit(f, FX, fX, p0)
    p = coef(fit)

    GSDist(F₀, quantile(dist, F₀), p..., dist)
end



function Base.show(io::IO, G::GSDist)
    a, g, k, y = params(G)
    println(io, "Generalized S-Distribution ($(G.dist))")
    println(io, " α: $a")
    println(io, " g: $g")
    println(io, " k: $k")
    print(io, " γ: $y")
end



params(G::GSDist) = (G.α, G.g, G.k, G.γ)



function quantile(D::GSDist, p::Real)
    q = _gsd_quantile(p, D.F₀, D.x₀, D.α, D.g, D.k, D.γ)
    if isnan(q)
        @debug "Unable to calculate the quantile of the GSDist. Falling back to the quantile of the underlying distribution."
        return quantile(D.dist, p)
    else
        return q
    end
end



function mean(G::GSDist)
    m = _gsd_mean(G.F₀, G.x₀, G.α, G.g, G.k, G.γ)
    if isnan(m)
        @debug "Unable to calculate the mean of the GSDist. Falling back to the mean of the underlying distribution."
        return mean(G.dist)
    else
        return m
    end
end



function var(G::GSDist)
    m1 = mean(G)
    m2 = try
        return _gsd_moment(G.F₀, G.x₀, G.α, G.g, G.k, G.γ, moment=2)
    catch y
        if isa(y, DomainError)
            @debug "Unable to calculate the variance of the GSDist. Falling back to the variance of the underlying distribution."
            return var(G.dist)
        end
    end
    return m2 - m1^2
end



std(G::GSDist) = sqrt(var(G))

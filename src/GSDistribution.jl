"""
    GSDistribution

Generalized S-Distribution.
"""
struct GSDistribution <: CUD
    F::UD
    F₀::Float64
    x₀::Float64
    α::Float64
    g::Float64
    k::Float64
    γ::Float64
end

function GSDistribution(D::UD, F₀::Real=0.5)
    F₀ < 0 || F₀ > 1 && throw(DomainError(F₀, "F₀ must be between 0 and 1"))
    F₀ = Float64(F₀)

    l,u = extrema(D)
    if isinf(l) l = quantile(D, 1e-6) end
    if isinf(u) u = quantile(D, 1 - 1e-6) end
    ql, qu = cdf.(D, (l,u))

    q = range(ql, qu, length=101)
    X = typeof(D) <: DUD ? Set(quantile.(D, q)) : quantile.(D, q)

    FX = cdf.(D, X)
    fX = typeof(D) <: DUD ? (cdf.(D, X.+1) - cdf.(D, X.-1)) / 2 : pdf.(D, X)
    
    f = (t, p) -> p[1] * t.^p[2] .* (1.0 .- t.^p[3]).^p[4]
    
    p0 = Float64[1, 0.5, 1, 0.5]
    fit = curve_fit(f, FX, fX, p0)
    p = coef(fit)

    GSDistribution(D, F₀, quantile(D, F₀), p...)
end


_q(p::Float64, F₀::Float64, x₀::Float64, α::Float64, g::Float64, k::Float64, γ::Float64) = x₀ + _beta_inc(F₀^k, p^k, (1-g)/k, 1-γ) / (α*k)
function Distributions.quantile(D::GSDistribution, p::Float64)
    q = _q(p, D.F₀, D.x₀, D.α, D.g, D.k, D.γ)
    if isnan(q)
        @warn "Unable to calculate the quantile of the GSDist. Falling back to the quantile of the underlying distribution."
        quantile(D.F, p)
    else
        q
    end
end
Distributions.quantile(D::GSDistribution, p::Real) = quantile(D, Float64(p))


function _mean(D::GSDistribution)
	z1 = D.F₀^D.k
	a1 = (1 - D.g) / D.k
	b1 = 1 - D.γ
	a2 = (2 - D.g) / D.k
	b2 = 1 - D.γ
	D.x₀ + (_beta_inc(z1, prevfloat(1.0), a1, b1) - _beta_inc(a2, b2)) * inv(D.α * D.k)
end
function Distributions.mean(D::GSDistribution)
    m = _mean(D)
    if isnan(m)
        @warn "Unable to calculate the mean of the GSDist. Falling back to the mean of the underlying distribution."
        mean(D.F)
    else
        m
    end
end


function _moment(D::GSDistribution, j::Int=1)
    x₀ = D.x₀
    z1 = D.F₀^D.k
    a = (1 - D.g) / D.k
    b = 1 - D.γ
    c = inv(D.α * D.k)
    
    f = q -> (x₀ + c*_beta_inc(z1, q^D.k, a, b))^j
    quadgk(f, 0, 1, atol=1e-4)[1]
end


function Distributions.var(D::GSDistribution)
    m1 = mean(D)
    try
        m2 = _moment(D, 2)
    catch y
        if isa(y, DomainError)
            @warn "Unable to calculate the variance of the GSDist. Falling back to the variance of the underlying distribution."
            return var(D.F)
        end
    end
    m2 - m1^2
end
Distributions.std(D::GSDistribution) = sqrt(var(D))


function _beta_inc(z1::Float64, z2::Float64, a::Float64, b::Float64)
    inv(a) * (z2^a * _₂F₁(a,1-b,a+1,z2) - z1^a * _₂F₁(a,1-b,a+1,z1))
end
_beta_inc(z1::Real, z2::Real, a::Real, b::Real) = _beta_inc(Float64.((z1,z2,a,b))...)
_beta_inc(x::Real, a::Real, b::Real) = _beta_inc(0.0, x, a, b)
_beta_inc(a::Real, b::Real) = _beta_inc(0.0, prevfloat(1.0), a, b)
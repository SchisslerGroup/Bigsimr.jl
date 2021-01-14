struct GSDistribution <: CUD
    F::UD
    F₀::Float64
    x₀::Float64
    α::Float64
    g::Float64
    k::Float64
    γ::Float64
end

function GSDistribution(D::UD)
    q = range(0.0001, 0.9999, length=10)
    X = quantile.(D, q)
    FX = cdf.(D, X)
    fX = typeof(D) <: DUD ? (cdf.(D, X.+1) - cdf.(D, X.-1)) / 2 : pdf.(D, X)
    
    f = (t, p) -> p[1] * t.^p[2] .* (1.0 .- t.^p[3]).^p[4]
    p0 = ones(Float64, 4)
    fit = curve_fit(f, FX, fX, p0)
    p = coef(fit)
    
    F₀ = 0.5
    GSDistribution(D, F₀, quantile(D, F₀), p...)
end

function Distributions.quantile(D::GSDistribution, p::Float64)
    D.x₀ + beta_inc_gen(D.F₀^D.k, p^D.k, (1-D.g)/D.k, 1-D.γ) / (D.α*D.k)
end
Distributions.mean(D::GSDistribution) = mean(D.F)
Distributions.std(D::GSDistribution) = std(D.F)

function beta_inc_gen(z1::Float64, z2::Float64, a::Float64, b::Float64)
    beta(a,b) * diff(cdf.(Beta(a,b), [z1, z2]))[1]
end

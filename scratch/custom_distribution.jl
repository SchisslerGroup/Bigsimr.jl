using Distributions
using StatsBase
using Statistics
import SpecialFunctions: gamma_inc


struct myPareto{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T
    myPareto{T}(α::T, θ::T) where {T} = new(α, θ)
end


function myPareto(α::T, θ::T; check_args=true) where {T<:Real}
    if check_args && (α ≤ 0 || θ ≤ 0)
        error("Both α and θ must be greater than 0")
    end
    return myPareto{T}(α, θ)
end

myPareto(α::Real, θ::Real) = myPareto(promote(α, θ)...)
myPareto(α::Integer, θ::Integer) = myPareto(float(α), float(θ))
myPareto(α::T) where {T <: Real} = myPareto(α, one(T))
myPareto() = myPareto(1.0, 1.0, check_args=false)

shape(d::Pareto) = d.α
scale(d::Pareto) = d.θ

StatsBase.params(d::myPareto) = (d.α, d.θ)

Statistics.mean(d::myPareto) = ((α, θ) = params(d); α ≤ 1 ? Inf : α*θ/(α-1))
function Statistics.var(d::myPareto)
    (α, θ) = params(d)
    α > 2 ? θ^2 * α / ((α - 1)^2 * (α - 2)) : T(Inf)
end

function Distributions.pdf(d::myPareto{T}, x::Real) where {T <: Real}
    (α, θ) = params(d)
    x ≥ θ ? α*θ^α / x^(α+1) : zero(T)
end

function Distributions.logpdf(d::myPareto{T}, x::Real) where {T <: Real}
    (α, θ) = params(d)
    x ≥ θ ? log(α) + α*log(θ) - (α+1) * log(x) : -T(Inf)
end

function Distributions.ccdf(d::myPareto{T}, x::Real) where {T <: Real}
    (α, θ) = params(d)
    x ≥ θ ? (θ/x)^α : one(T)
end

function Distributions.logccdf(d::myPareto{T}, x::Real) where {T <: Real}
    (α, θ) = params(d)
    x ≥ θ ? α * log(θ / x) : zero(T)
end

Distributions.cdf(d::myPareto, x::Real) = 1 - ccdf(d, x)
Distributions.logcdf(d::myPareto, x::Real) = log1p(-ccdf(d, x))
Distributions.quantile(d::myPareto, p::Real) = d.θ / (1 - p)^(1 / d.α)
Distributions.cquantile(d::myPareto, p::Real) = quantile(d, 1 - p)

function Distributions.skewness(d::myPareto{T}) where {T <: Real}
    (α, θ) = params(d)
    α > 3 ? (2*(1+α)/(α-3))*sqrt((α-2)/α) : T(NaN)
end

function Distributions.kurtosis(d::myPareto{T}) where {T <: Real}
    (α, θ) = params(d)
    α > 4 ? 6*(α^3 + α^2 - 6α - 2) / (α*(α - 3)*(α - 4)) : T(NaN)
end

Distributions.entropy(d::myPareto) = log(d.θ / d.α) + 1 + 1 / d.α

function Base.minimum(d::myPareto{T}) where {T <: Real}
    d.θ
end

function Base.maximum(d::myPareto{T}) where {T <: Real}
    T(Inf)
end

function Distributions.fit_mle(::Type{<:myPareto}, x::AbstractArray{T}) where {T <: Real}
    θ = minimum(x)
    n = length(x)
    α = n / sum(log.(x) .- log(θ))
    Pareto(α, θ)
end

D = myPareto(5, π)
params(D)
mean(D)
var(D)
std(D)
pdf.(D, 0:0.5:5)
cdf.(D, 0:0.5:10)
ccdf.(D, 0:0.5:10)
logpdf.(D, 0:0.5:10)
quantile(D, 0)
cquantile(D, 0)
minimum(D)
maximum(D)

insupport(D, 5.6)
insupport(D, 2)

x = rand(D, 1000)
fit_mle(myPareto, x)

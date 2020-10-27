using MvSim
using Distributions

struct MvDistribution
    R::Matrix{<:Real}
    margins::Vector{<:UnivariateDistribution}
    C::Type{<:Correlation}
end

function Base.rand(D::MvDistribution, n::Int)
    rvec(n, D.margins, D.R)
end

m = [
    Beta(2, 3), 
    Normal(5, 2.2), 
    Binomial(2, 0.2), 
    Binomial(20, 0.2),
    Uniform(-6, 12.4)
]
r = cor_randPD(Float64, length(m), 1.0)
D = MvDistribution(r, m, Pearson)

rand(D, 10)
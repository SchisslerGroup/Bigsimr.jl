"""
    _randn(n::Int, d::Int)

Fast parallel generation of standard normal samples.
"""
function _randn(n::Int, d::Int)
    Z = SharedMatrix{Float64}(n, d)
    @inbounds @threads for i in eachindex(Z)
        Z[i] = randn(Float64)
    end
    sdata(Z)
end


"""
    _rmvn(n::Int, ρ::Matrix{Float64})

Fast parallel generation of multivariate standard normal samples.
"""
function _rmvn(n::Int, ρ::Matrix{Float64})
	Z = _randn(n, size(ρ, 1))
	C = cholesky(ρ)
	Z * C.U
end


"""
    normal_to_margin(d::UnivariateDistribution, x::Float64)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
normal_to_margin(d::UD, x::Float64) = quantile(d, cdf(Normal(0,1), x))
normal_to_margin(d::UD, X::AbstractVecOrMat) = normal_to_margin.(d, X)


"""
    rvec(n, ρ, margins)

Generate samples for a list of marginal distributions and a correaltion structure.
"""
function rvec end

function rvec(n::Int, ρ::Matrix{Float64}, margins::Vector{<:UD})
    d = length(margins)
    Z = _rmvn(n, ρ)
    @threads for i in 1:d
        @inbounds Z[:,i] = normal_to_margin(margins[i], Z[:,i])
    end
    return Z
end


"""
    rand(D::MvDistribution, n::Int)

More general wrapper for `rvec`.
"""
function rand(D::MvDistribution, n::Int)
    rvec(n, cor(D), margins(D))
end
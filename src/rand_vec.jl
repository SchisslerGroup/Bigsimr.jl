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
    rmvn(n[, μ], Σ)

Fast parrallel generation of multivariate normal samples.
"""
function rmvn end
rmvn(n::Int, μ::Vector{Float64}, Σ::Matrix{Float64}) = μ' .+ _rmvn(n, Σ)
rmvn(n::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real}) = rmvn(Int(n), Vector{Float64}(μ), Matrix{Float64}(Σ))
function rmvn(n::Real, Σ::Matrix{<:Real})
    d = size(Σ, 2)
    rmvn(n, ones(Float64, d), Σ)
end


"""
    normal_to_margin(d::UnivariateDistribution, x::Float64)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
normal_to_margin(d::UD, x::Float64) = quantile(d, _normcdf(x))
normal_to_margin(d::UD, V::Vector{Float64}) = normal_to_margin.(d, V)
normal_to_margin(d::UD, X::Matrix{Float64}) = normal_to_margin.(d, X)


"""
    rvec(n::Int, ρ::Matrix{Float64}, margins::Vector{<:UnivariateDistribution})

Generate samples for a list of marginal distributions and a correaltion structure.
"""
function rvec(n::Int, ρ::Matrix{Float64}, margins::Vector{<:UD})
    d = length(margins)
    r,s = size(ρ)

    !(r == s == d) && throw(DimensionMismatch("The number of margins must match the size of the correlation matrix."))
    !iscorrelation(ρ) && throw(ValidCorrelationError())

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
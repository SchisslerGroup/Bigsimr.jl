function _rmvn(n::Int, ρ::Matrix{Float64})
	Z = _randn(n, size(ρ, 1))
	C = cholesky(ρ)
	Z * C.U
end

function _randn(n::Int, d::Int)
	Z = SharedMatrix{Float64}(n, d)
	Threads.@threads for i in 1:n*d
		Z[i] = randn(Float64)
	end
	sdata(Z)
end

"""
    rvec(n, margins, ρ)

Generate samples for a list of marginal distributions and a correaltion structure.
"""
function rvec end

function rvec(n::Int, margins::Vector{<:UD}, ρ::Matrix{<:Real})
    d = length(margins)
    Z = _rmvn(n, ρ)
    @threads for i in 1:d
        @inbounds Z[:,i] = normal_to_margin(margins[i], Z[:,i])
    end
    return Matrix{eltype(Z)}(Z)
end


"""
    Base.rand(D::MvDistribution, n::Int)

More general wrapper for `rvec`.
"""
function Base.rand(D::MvDistribution, n::Int)
    rvec(n, D.margins, D.R)
end
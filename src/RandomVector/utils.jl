for T in (Float64, Float32, Float16)
    @eval function _randn(::Type{$T}, n::Int, d::Int)
        Z = SharedMatrix{$T}(n, d)
        @inbounds @threads for i in eachindex(Z)
            Z[i] = randn($T)
        end
        sdata(Z)
    end
    @eval _randn(::Type{$T}, n::Real, d::Real) = _randn($T, Int(n), Int(d))
end
_randn(n::Real, d::Real) = _randn(Float64, Int(n), Int(d))


for T in (Float64, Float32, Float16)
    @eval function _rmvn(n::Int, ρ::Matrix{$T})
        Z = _randn($T, n, size(ρ, 1))
        C = cholesky(ρ)
        Z * Matrix{$T}(C.U)
    end
end
_rmvn(n::Int, ρ::Float64) = _rmvn(n, [1.0 ρ; ρ 1.0])


normal_to_margin(d::UD, x::Float64) = quantile(d, _normcdf(x))
normal_to_margin(d::UD, x::Real) = normal_to_margin(d, Float64(x))
normal_to_margin(d::UD, A::Array{<:Real, N}) where N = normal_to_margin.(d, Array{Float64, N}(A))

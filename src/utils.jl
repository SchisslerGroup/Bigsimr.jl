"""
    iscorrelation(X)

Check if the given matrix is a valid correlation matrix.
"""
function iscorrelation(X)
    issymmetric(X)     || return false
    all(diag(X) .== 1) || return false
    all(-1 .≤ X .≤ 1)  || return false
    isposdef(X)        || return false

    return true
end



#--------------------------------------------------------------Normal PDF, CDF, Inverse CDF
_normpdf(x::Real) = exp(-abs2(x) / 2) * invsqrt2π
_normpdf(x::Float16) = Float16(_normpdf(Float32(x)))

_normcdf(x::Real) = erfc(-x * invsqrt2) / 2
_normcdf(x::Float16) = Float16(_normcdf(Float32(x)))

_norminvcdf(x::Real) = -1 * erfcinv(2 * x) * sqrt2
_norminvcdf(x::Float16) = Float16(_norminvcdf(Float32(x)))

_norm2margin(D::UD, x::Real) = quantile(D, _normcdf(x))
_norm2margin(D::UD, A::AbstractArray{T,N}) where {T<:Real,N} = _norm2margin.(Ref(D), A)



#---------------------------------------------------------------------Random Normal Samples
function _randn(::Type{T}, n::Int, d::Int) where {T<:Real}
    Z = SharedMatrix{T}(n, d)
    @inbounds @threads for i in eachindex(Z)
        Z[i] = randn(T)
    end
    return sdata(Z)
end

_randn(T, n, d) = _randn(T, Int(n), Int(d))

# This method exists for use in R where all numbers are type Double
_randn(n, d) = _randn(Float64, Int(n), Int(d))




#--------------------------------------------------------Random Multivariate Normal Samples
function _rmvn(n::Int, rho::AbstractMatrix{T}) where {T<:Real}
    Z = _randn(T, n, size(rho, 1))
    C = cholesky(rho)
    return Z * C.U
end

_rmvn(n::Int, rho::Real) = _rmvn(n, [1 rho; rho 1])

# This method exists for use in R where all numbers are type Double
_rmvn(n::Real, rho) = _rmvn(Int(n), rho)

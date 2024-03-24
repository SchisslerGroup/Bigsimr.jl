module BigsimrBase


using Distributions: UnivariateDistribution as UD
using Distributions: quantile
using LinearAlgebra: cholesky, rmul!
using SharedArrays
using StatsFuns: normcdf


include("cortype.jl")


"""
    _cor(cortype, args...; kwargs...)

Base cor method for abstraction.
"""
function _cor(::CorType, args...; kwargs...) end


# transforms a standard normal sample to the specified margin (NORTA step)
norm2margin(D::UD, x::Real) = quantile(D, normcdf(x))
norm2margin(D::UD, A::AbstractArray{T,N}) where {T<:Real,N} = norm2margin.(Ref(D), A)


# generates random normal samples in parallel
function randn_shared(::Type{T}, n::Int, d::Int) where {T<:Real}
    n * d < 1_000_000 && return SharedMatrix(randn(T, n, d))

    Z = SharedMatrix{T}(n, d)

    Base.Threads.@threads for i in 1:d
        @inbounds @view(Z[:,i]) .= randn(T, n)
    end

    return Z
end

randn_shared(n::Int, d::Int) = randn_shared(Float64, n, d)


# generates random multivariate normal samples in parallel
function rmvn_shared(n::Int, rho::AbstractMatrix{T}) where {T<:Real}
    Z = randn_shared(T, n, size(rho, 1))
    C = cholesky(rho)
    rmul!(Z, C.U)
    return Z
end

rmvn_shared(n::Int, rho::Real) = rmvn_shared(n, [1 rho; rho 1])


# equivalent to IterTools.subsets(1:d, Val(2)), but allocates for all elements
function idx_subsets2(d::Int)
    n = d * (d - 1) ÷ 2
    xs = Vector{Tuple}(undef, n)

    k = 1
    for i = 1:d-1
        for j = i+1:d
            xs[k] = (i,j)
            k += 1
        end
    end

    return xs
end


# copies the upper part of a square matrix to the lower (not including the diagonal)
function make_symmetric!(X::AbstractMatrix{T}) where {T}
    m, n = size(X)
    m == n || throw(DimensionMismatch("Input matrix must be square"))

    for i = 1:n-1
        for j = i+1:n
            X[j,i] = X[i,j]
        end
    end

    return X
end


# sets the diagonal elements of a square matrix to 1
function set_diag1!(X::AbstractMatrix{T}) where {T}
    m, n = size(X)
    m == n || throw(DimensionMismatch("Input matrix must be square"))

    for i in 1:n
        @inbounds X[i,i] = one(T)
    end

    return X
end


# constrains a value between ±1
clampcor(x::Real) = clamp(x, -one(x), one(x))


# convenience function for getting a negative definite matrix for testing
function make_negdef_matrix(T=Float64)
    r = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]

    return convert(AbstractMatrix{T}, r)
end


end

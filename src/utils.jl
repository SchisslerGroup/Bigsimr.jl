# Type promotion for setdiag()
function promote(A::Matrix{T}, x::S) where {T<:Real, S<:Real}
    TS = promote_type(T, S)
    (Matrix{TS}(A), TS(x))
end


"""
    setdiag(A::Matrix{T}, x::S) where {T<:Real, S<:Real}

Set the diagonal elements of a Matrix to a value. Return the new matrix.
"""
function setdiag(A::Matrix{<:Real}, x::Real)
    A, x = promote(A, x)
    @inbounds A[diagind(A)] .= x
    A
end

function setdiag!(A::Matrix{T}, x::T) where {T<:Real}
    @inbounds A[diagind(A)] .= x
end


"""
    normal_to_margin(d::UnivariateDistribution, x)

Convert samples from a standard normal distribution to a given marginal distribution.
"""
function normal_to_margin(d::UD, x)
    quantile.(d, cdf.(Normal(0,1), x))
end

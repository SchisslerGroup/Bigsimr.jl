struct PDCorMat{T<:Real, S<:AbstractMatrix} <: AbstractPDMat{T}
    dim::Int
    mat::S
    chol::Cholesky{T,S}
    cortype::Type{<:Correlation}

    PDCorMat{T,S}(d::Int, m::AbstractMatrix{T}, c::Cholesky{T,S}, t::Type{<:Correlation}) where {T,S} = new{T,S}(d,m,c,t)
end

function PDCorMat(mat::AbstractMatrix, cortype::Type{<:Correlation})
    d = size(mat, 1)

    if !iscorrelation(mat)
        mat .= cor_nearPD(mat)
    end
    chol = cholesky(mat)
    PDCorMat{eltype(mat), typeof(mat)}(d, mat, chol, cortype)
end
PDCorMat(mat::AbstractMatrix) = PDCorMat(mat, Pearson)


dim(a::PDCorMat) = a.dim
cortype(a::PDCorMat) = a.cortype


pdadd(m::PDCorMat, a::PDCorMat, c::Real) = m + a * c
function pdadd!(r::Matrix, m::Matrix, a::PDCorMat, c::Real)
    r .= m + a * c
    return r
end


quad(a::PDCorMat, x::Vector) = dot(x, a * x)
function quad!(r::AbstractArray, a::PDCorMat, x::Matrix)
    for i in 1:size(x, 2)
        r[i] = quad(a, x[:,i])
    end
    return r
end


invquad(a::PDCorMat, x::Vector) = dot(x, a \ x)
function invquad!(r::AbstractArray, a::PDCorMat, x::Matrix)
    for i in 1:size(x, 2)
        r[i] = invquad(a, x[:,i])
    end
    return r
end


function X_A_Xt(a::PDCorMat, x::Matrix)
    z = x * a.chol.L
    z * transpose(z)
end
function Xt_A_X(a::PDCorMat, x::Matrix)
    z = transpose(x) * a.chol.L
    z * transpose(z)
end
function X_invA_Xt(a::PDCorMat, x::Matrix)
    z = a.chol \ transpose(x)
    x * z
end
function Xt_invA_X(a::PDCorMat, x::Matrix)
    z = a.chol \ x
    transpose(x) * z
end


function whiten!(r::VecOrMat, a::PDCorMat, x::VecOrMat)
    copyto!(r, x)
    rdiv!(r, a.chol.U)
end
unwhiten!(r::VecOrMat, a::PDCorMat, x::VecOrMat) = mul!(r, x, a.chol.U)


diag(a::PDCorMat)   = ones(eltype(a), a.dim)
inv(a::PDCorMat)    = PDMat(inv(a.chol))
logdet(a::PDCorMat) = logdet(a.chol)


Base.Matrix(a::PDCorMat) = Matrix(a.mat)
Base.size(a::PDCorMat) = size(a.mat)
Base.getindex(a::PDCorMat, i::Int) = getindex(a.mat, i)
Base.getindex(a::PDCorMat, I::Vararg{Int, N}) where {N} = getindex(a.mat, I...)

Base.:*(a::PDCorMat, c::T) where {T<:Real} = a.mat * c
Base.:*(a::PDCorMat, x::VecOrMat) = a.mat * x
Base.:\(a::PDCorMat, x::VecOrMat) = a.chol \ x

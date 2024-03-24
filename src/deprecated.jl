function Statistics.cor(x::AbstractVecOrMat, y::AbstractVecOrMat, cortype::CorType)
    @warn "The syntax 'cor(x, y, cortype)' is now deprecated. Please use 'cor(cortype, x, y)' instead." maxlog=1
    return cor(cortype, x, y)
end

function Statistics.cor(x::AbstractVecOrMat, cortype::CorType)
    @warn "The syntax 'cor(x, cortype)' is now deprecated. Please use 'cor(cortype, x)' instead." maxlog=1
    return cor(cortype, x)
end

function cor_fast(X::AbstractMatrix, cortype::CorType=Pearson)
    @warn "The syntax 'cor_fast(X, cortype)' is now deprecated. Please use 'cor_fast(cortype, X)' instead." maxlog=1
    return cor_fast(cortype, X)
end

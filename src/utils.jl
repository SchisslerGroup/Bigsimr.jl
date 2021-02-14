function iscorrelation(X::Matrix{<:AbstractFloat})
    all([
        isposdef(X),
        issymmetric(X),
        all(diag(X) .== one(eltype(X))),
        all(-one(eltype(X)) .≤ X .≤ one(eltype(X)))
    ])
end


for T in (Float64, Float32)
    @eval _normpdf(x::$T) = exp(-abs2(x)/2) * $T(invsqrt2π)
    @eval _normcdf(x::$T) = erfc(-x * $T(invsqrt2)) / 2
    @eval _norminvcdf(x::$T) = -$T(sqrt2) * erfcinv(2x)
end
for F in (:_normpdf, :_normcdf, :_norminvcdf)
    @eval $F(x::Float16) = Float16($F(Float32(x)))
end
for T in (Int16, Int32, Int64)
    for F in (:_normpdf, :_normcdf, :_norminvcdf)
        @eval $F(x::$T) = $F(Float64(x))
    end
end
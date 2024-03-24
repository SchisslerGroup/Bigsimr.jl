"""
    cor_nearPD(X)

Return the nearest positive definite correlation matrix to `X`.

See also: [`cor_nearPSD`](@ref), [`cor_fastPD`](@ref)
"""
cor_nearPD(X) = nearest_cor(X)

"""
    cor_nearPD!(X)

Same as [`cor_nearPD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPSD!`](@ref), [`cor_fastPD!`](@ref)
"""
cor_nearPD!(X) = nearest_cor!(X)

"""
    cor_nearPSD(X)

Return the nearest positive [semi-] definite correlation matrix to `X`.

See also: [`cor_nearPD`](@ref), [`cor_fastPD`](@ref)
"""
function cor_nearPSD(X)
    sol = solve(
        NCMProblem(X),
        nothing;
        alias_A=false,
        fix_sym=true,
        convert_f16=true,
        ensure_pd=false
    )

    return sol.X
end

"""
    cor_nearPSD!(X)

Same as [`cor_nearPSD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPD!`](@ref), [`cor_fastPD!`](@ref)
"""
function cor_nearPSD!(X)
    sol = solve(
        NCMProblem(X), nothing; alias_A=true, fix_sym=true, convert_f16=true, ensure_pd=true
    )

    copyto!(X, sol.X)
    return X
end

"""
    cor_fastPD(X[, tau])

Return a positive definite correlation matrix that is close to `X`. `tau` is a
tuning parameter that controls the minimum eigenvalue of the resulting matrix.
`τ` can be set to zero if only a positive semidefinite matrix is needed.

See also: [`cor_nearPD`](@ref), [`cor_nearPSD`](@ref)
"""
cor_fastPD(X, tau) = nearest_cor(X, DirectProjection(tau))
cor_fastPD(X) = nearest_cor(X, DirectProjection)

"""
    cor_fastPD!(X[, tau])

Same as [`cor_fastPD`](@ref), but saves space by overwriting the input `X` instead of
creating a copy.

See also: [`cor_nearPD!`](@ref), [`cor_nearPSD!`](@ref)
"""
cor_fastPD!(X, tau) = nearest_cor!(X, DirectProjection(tau))
cor_fastPD!(X) = nearest_cor!(X, DirectProjection)

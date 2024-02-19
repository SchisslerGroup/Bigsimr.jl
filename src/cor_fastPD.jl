function _fast_pca!(X::AbstractMatrix{T}, λ::AbstractVector{T}, P::AbstractMatrix{T}, n::Int) where {T}
    r = sum(λ .> 0)
    r == n && return X
    s = n - r

    r == 0 && return fill!(X, zero(T))

    if r == 1
        X .= (λ[1] * λ[1]) * (P[:,1] * P[:,1]')
        return X
    end

    if r ≤ s
        P₁   = @view P[:, 1:r]
        λ₁   = sqrt.(λ[1:r])
        P₁λ₁ = P₁ .* λ₁'
        X   .= P₁λ₁ * P₁λ₁'
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂'
        X   .= X .+ P₂λ₂ * P₂λ₂'
    end

    return X
end



"""
    cor_fastPD!(R::AbstractMatrix{<:Real}, tau=1e-6)

Same as [`cor_fastPD`](@ref), but saves space by overwriting the input `R` instead of
creating a copy.

See also: [`cor_fastPD`](@ref), [`cor_nearPD`](@ref)

# Examples
```jldoctest
julia> using LinearAlgebra: isposdef

julia> r = [1.00 0.82 0.56 0.44; 0.82 1.00 0.28 0.85; 0.56 0.28 1.00 0.22; 0.44 0.85 0.22 1.00]
4×4 Matrix{Float64}:
 1.0   0.82  0.56  0.44
 0.82  1.0   0.28  0.85
 0.56  0.28  1.0   0.22
 0.44  0.85  0.22  1.0

julia> isposdef(r)
false

julia> cor_fastPD!(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(r)
true
```
"""
function cor_fastPD!(R::AbstractMatrix{T}, tau=1e-6) where {T<:Real}
    n  = size(R, 1)
    tau  = max(eps(T), tau)

    R .= Symmetric(R, :U)
    R[diagind(R)] .= (one(T) - tau)

    λ, P = eigen(R)
    λ   .= reverse(λ)
    P   .= reverse(P, dims=2)

    _fast_pca!(R, λ, P, n)

    R[diagind(R)] .+= tau
    return cov2cor!(R)
end



"""
    cor_fastPD(R, tau=1e-6)

Return a positive definite correlation matrix that is close to `R`. `τ` is a
tuning parameter that controls the minimum eigenvalue of the resulting matrix.
`τ` can be set to zero if only a positive semidefinite matrix is needed.

See also: [`cor_fastPD!`](@ref), [`cor_nearPD`](@ref)

# Examples
```jldoctest
julia> import LinearAlgebra: isposdef

julia> r = [1.00 0.82 0.56 0.44; 0.82 1.00 0.28 0.85; 0.56 0.28 1.00 0.22; 0.44 0.85 0.22 1.00]
4×4 Matrix{Float64}:
 1.0   0.82  0.56  0.44
 0.82  1.0   0.28  0.85
 0.56  0.28  1.0   0.22
 0.44  0.85  0.22  1.0

julia> isposdef(r)
false

julia> p = cor_fastPD(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(p)
true
```
"""
cor_fastPD(R::AbstractMatrix{<:Real}, tau=1e-6) = cor_fastPD!(copy(R), tau)

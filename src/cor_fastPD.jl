function fast_pca!(X::Matrix{T}, λ::Vector{T}, P::Matrix{T}, n::Int) where T<:AbstractFloat
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        X .= zeros(T, n, n)
    elseif r == n
        return nothing
    elseif r == 1 
        X .= (λ[1] * λ[1]) * (P[:,1] * P[:,1]')   
    elseif r ≤ s
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
    nothing
end


"""
    cor_fastPD!(R::Matrix{<:AbstractFloat}[, τ=1e-6])

Same as [`cor_fastPD`](@ref), but saves space by overwriting the input `R`
instead of creating a copy.

See also: [`cor_fastPD`](@ref), [`cor_nearPD`](@ref)

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

julia> cor_fastPD!(r)


julia> isposdef(r)
true
```
"""
function cor_fastPD!(R::Matrix{<:AbstractFloat}, τ=1e-6)
    n  = size(R, 1)
    τ  = max(eps(eltype(R)), τ)
    
    R .= Symmetric(R, :U)
    R[diagind(R)] .= (one(eltype(R)) - τ)

    λ, P = eigen(R)
    λ   .= reverse(λ)
    P   .= reverse(P, dims=2)

    fast_pca!(R, λ, P, n)

    R[diagind(R)] .+= τ
    R .= cov2cor(R)
    nothing
end


"""
    cor_fastPD(R::Matrix{<:AbstractFloat}[, τ=1e-6])

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
function cor_fastPD(R::Matrix{<:AbstractFloat}, τ=1e-6)
    X = copy(R)
    cor_fastPD!(X, τ)
    X
end
function cor_fastPD(R::Matrix{Float16}, τ=1e-6)
    X = Matrix{Float32}(R)
    cor_fastPD!(X, τ)
    Matrix{Float16}(X)
end

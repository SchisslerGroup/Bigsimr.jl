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
        P₁λ₁ = P₁ .* λ₁' # each row of P₁ times λ₁
        X .= P₁λ₁ * P₁λ₁'
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂' # each row of P₂ times λ₂
        X .= X .+ P₂λ₂ * P₂λ₂'
    end
end

function cor_fastPD!(R::Matrix{<:AbstractFloat}, τ=1e-6)
    n = size(R, 1)
    R .= Symmetric(R, :U) # [n,n]
    R[diagind(R)] .= (one(eltype(R)) - τ)

    λ, P = eigen(R)
    λ   .= reverse(λ)
    P   .= reverse(P, dims=2)

    fast_pca!(R, λ, P, n)

    R[diagind(R)] .+= τ
    R .= cov2cor(R)
end
cor_fastPD(R::Matrix{<:AbstractFloat}, τ=1e-6) = cor_fastPD!(copy(R), τ)
"""
    npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
"""
function npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)

    if r == 0
        return (zero(Float64), zeros(Float64, n))
    else
        λ[λ .< 0] .= zero(Float64)
        Fy = vec(sum((P .* λ') .* P, dims=2))
        f  = 0.5 * sum(λ.^2) - sum(b₀ .* y)
        return (f, Fy)
    end
end


"""
    npd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
"""
function npd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
    r = sum(λ .> 0)
    s = n - r

    r == 0 && return zeros(Float64, n, n)
    r == 1 && return (λ[1] * λ[1]) * (P[:,1] * P[:,1]')
    r == n && return X
    
    if r ≤ s
        P₁   = @view P[:, 1:r]
        λ₁   = sqrt.(λ[1:r])
        P₁λ₁ = P₁ .* λ₁' # each row of P₁ times λ₁
        return P₁λ₁ * P₁λ₁'
    else
        P₂   = @view P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂' # each row of P₂ times λ₂
        return X .+ P₂λ₂ * P₂λ₂'
    end
end


"""
    npd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)

Pre- Conjugate Gradient method.
"""
function npd_pre_cg(
    b::Vector{Float64}, 
    c::Vector{Float64}, 
    Ω₀::Matrix{Float64}, 
    P::Matrix{Float64}, 
    ϵ::Float64, 
    N::Int, 
    n::Int)

    ϵ_b = ϵ * norm2(b)

    r   = copy(b)
    z   = r ./ c
    d   = copy(z)
    rz1 = sum(r .* z)
    rz2 = one(Float64)
    p   = zeros(Float64, n)

    for k in 1:N
        if k > 1
            d = z + d * (rz1 / rz2)
        end

        w = npd_jacobian(d, Ω₀, P, n)

        denom = sum(d .* w)
        normr = norm2(r)
        
        denom ≤ 0 && return d / norm2(d)
        
        α = rz1 / denom
        p += α*d
        r -= α*w
        
        norm2(r) ≤ ϵ_b && return p
        
        z = r ./ c
        rz2, rz1 = copy(rz1), sum(r .* z)
    end
    
    return p
end


"""
    npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
"""
function npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
    r, s = size(Ω₀)

    r == 0 || r == n && return ones(Float64, n)

    H  = (P .* P)'
    H₁ = @view H[1:r,:]
    H₂ = @view H[r+1:n,:]

    if r < s
        H12 = H₁' * Ω₀
        c   = sum(H₁, dims=1)'.^2 + 2.0 * sum(H12 .* H₂', dims=2)
    else
        H12 = (1.0 .- Ω₀) * H₂
        c   = sum(H, dims=1)'.^2 - sum(H₂, dims=1)'.^2 - 2.0 * sum(H₁ .* H12, dims=1)'
    end

    c[c .< 1e-8] .= 1e-8
    return c
end


"""
    npd_set_omega(λ::Vector{Float64}, n::Int)
"""
function npd_set_omega(λ::Vector{Float64}, n::Int)
    r = sum(λ .> 0)
    s = n - r

    r == 0 && return zeros(Float64, 0, 0)
    r == n && return ones(Float64, n, n)
    
    M = zeros(Float64, r, s)
    λᵣ = @view λ[1:r]
    λₛ = @view λ[r+1:n]
    for j = 1:s, i = 1:r
        @inbounds M[i,j] = λᵣ[i] / (λᵣ[i] - λₛ[j])
    end

    return M
end


"""
    npd_jacobian(x, Ω₀, P, n; PERTURBATION=1e-9)
"""
function npd_jacobian(
    x::Vector{Float64}, 
    Ω₀::Matrix{Float64}, 
    P::Matrix{Float64}, 
    n::Int; 
    PERTURBATION::Float64=1e-9)

    r, s = size(Ω₀)

    r == 0 && return zeros(Float64, n)
    r == n && return x .* (1.0 + PERTURBATION)

    P₁ = @view P[:, 1:r]
    P₂ = @view P[:, r+1:n]

    if r < s
        H₁ = diagm(x) * P₁
        Ω  = Ω₀ .* (H₁' * P₂)

        HT₁ = P₁ * P₁' * H₁ + P₂ * Ω'
        HT₂ = P₁ * Ω

        return sum(P .* [HT₁ HT₂], dims=2) + x .* PERTURBATION
    else
        H₂ = diagm(x) * P₂
        Ω  = (1.0 .- Ω₀) .* (P₁' * H₂)

        HT₁ = P₂ * Ω'
        HT₂ = P₂ * H₂' * P₂ + P₁ * Ω

        return x .* (1.0 + PERTURBATION) - sum(P .* [HT₁ HT₂], dims=2)
    end
end


"""
    cor_nearPD(R::Matrix{Float64};
        τ::Float64=1e-5,
        iter_outer::Int=200,
        iter_inner::Int=20,
        N::Int=200,
        err_tol::Float64=1e-6,
        precg_err_tol::Float64=1e-2,
        newton_err_tol::Float64=1e-4)

Compute the nearest positive definite correlation matrix given a symmetric
correlation matrix `R`. This algorithm is based off of work by Qi and Sun 2006.
Matlab, C, R, and Python code can be found [on Sun's page](https://www.polyu.edu.hk/ama/profile/dfsun/index.html#Codes).
The algorithm has also been implemented in Fortran in the NAG library.

# Arguments
- `τ::Float64`: a [small] nonnegative number used to enforce a minimum eigenvalue.
- `err_tol::Float64`: the error tolerance for the stopping condition.

# Examples
```julia
import LinearAlgebra: eigvals
# Define a negative definite correlation matrix
ρ = [1.00 0.82 0.56 0.44
     0.82 1.00 0.28 0.85
     0.56 0.28 1.00 0.22
     0.44 0.85 0.22 1.00]
eigvals(ρ)

r = cor_nearPD(ρ)
eigvals(r)
```
"""
function cor_nearPD(R::Matrix{Float64}; # [n,n]
    τ::Float64=1e-5,
    iter_outer::Int=200,
    iter_inner::Int=20,
    N::Int=200,
    err_tol::Float64=1e-6,
    precg_err_tol::Float64=1e-2,
    newton_err_tol::Float64=1e-4)

    n = size(R, 1)

    # Make R symmetric
    R .= 0.5 .* (R + R') # [n,n]

    b = ones(Float64, n)
    if τ > 0
        b .-= τ
        R[diagind(r)] .-= τ
    end
    b₀ = copy(b)

    y    = zeros(Float64, n)  # [n,1]
    X    = copy(R)            # [n,n]
    λ, P = eigen(X)           # [n,1], [n,n]
    λ    = reverse(λ)         # [n,1]
    P    = reverse(P, dims=2) # [n,n]

    f₀, Fy = npd_gradient(y, λ, P, b₀, n) # [1], [n,1]
    f      = f₀ # [1]
    b     .= b₀ - Fy # [n,1]

    Ω₀ = npd_set_omega(λ, n) # [n,n] or [r,s]
    x₀ = copy(y) # [n,1]

    X       .= npd_pca(X, λ, P, n) # [n,n]
    val_R    = 0.5 * norm2(R)^2
    val_dual = val_R - f₀
    val_obj  = 0.5 * norm2(X - R)^2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

    normb  = norm2(b)
    normb0 = norm2(b₀) + 1
    Δnb    = normb / normb0

    k = 0
    c = zeros(Float64, n)
    d = zeros(Float64, n)
    while (gap > err_tol) && (Δnb > err_tol) && (k < iter_outer)
        c .= npd_precond_matrix(Ω₀, P, n)                 # [n,1]
        d .= npd_pre_cg(b, c, Ω₀, P, precg_err_tol, N, n) # [n,1]

        slope = sum((Fy - b₀) .* d) # [1]

        y    .= x₀ + d                       # [n,1]
        X    .= R + diagm(y)                 # [n,n]
        λ, P  = eigen(X)                     # [n,1], [n,n]
        λ    .= reverse(λ)                   # [n,1]
        P    .= reverse(P, dims=2)           # [n,n]
        f, Fy = npd_gradient(y, λ, P, b₀, n) # [1], [n,1]

        k_inner = 0
        while (k_inner ≤ iter_inner) && (f > f₀ + newton_err_tol * slope * 0.5^k_inner + 1e-6)
            k_inner += 1
            y    .= x₀ + d * 0.5^k_inner         # [n,1]
            X    .= R + diagm(y)                 # [n,n]
            λ, P  = eigen(X)                     # [n,1], [n,n]
            λ    .= reverse(λ)                   # [n,1], [n,n]
            P    .= reverse(P, dims=2)           # [n,n]
            f, Fy = npd_gradient(y, λ, P, b₀, n) # [1], [n,1]
        end

        x₀  = copy(y) # [n,1]
        f₀  = f

        X       .= npd_pca(X, λ, P, n) # [n,n]
        val_dual = val_R - f₀
        val_obj  = 0.5 * norm2(X - R)^2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b        = b₀ - Fy
        normb    = norm2(b)
        Δnb      = normb / normb0

        Ω₀ = npd_set_omega(λ, n) # [n,n] or [r,s]

        k += 1
    end

    X[diagind(X)] .+= τ
    return cov2cor(X)
end

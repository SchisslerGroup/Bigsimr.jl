"""
    npsd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64})
"""
function npsd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)

    if r == 0
        return (zero(Float64), zeros(Float64, n))
    else
        λ[λ .< 0] .= 0
        # 1) M[:,j] = P[:,j] * λ[j]
        # 2) M = M .* P
        # 3) Fy = rowsums(M)
        Fy = vec(sum((P .* λ') .* P, dims=2))
        f  = 0.5 * sum(λ.^2) - sum(b₀ .* y)
        return (f, Fy)
    end
end


"""
    npsd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64})
"""
function npsd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        return zeros(Float64, n, n)
    elseif r == 1
        return λ[1].^2 * (P[:,1] * P[:,1]')
    elseif r == n
        return X
    elseif r ≤ s
        P₁   = P[:, 1:r]
        λ₁   = sqrt.(λ[1:r])
        P₁λ₁ = P₁ .* λ₁'
        return P₁λ₁ * P₁λ₁'
    else
        P₂   = P[:, (r+1):n]
        λ₂   = sqrt.(-λ[(r+1):n])
        P₂λ₂ = P₂ .* λ₂'
        return X .+ P₂λ₂ * P₂λ₂'
    end
end


"""
    npsd_pre_cg(b, c, Ω₀, P, precg_err_tol, N)

Pre- Conjugate Gradient method.
"""
function npsd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)
    ϵ_b = ϵ * norm2(b)

    r   = copy(b)
    z   = r ./ c
    rz1 = sum(r .* z)
    rz2 = one(Float64)

    p  = zeros(Float64, n)
    d  = copy(z)

    w = Vector{Float64}(undef, n)
    for k in 1:N
        if k > 1
            @. d = z + d * (rz1 / rz2)
        end

        w .= npsd_jacobian(d, Ω₀, P)

        denom = sum(d .* w)
        normr = norm2(r)
        if denom ≤ 0
            return d / norm2(d)
        else
            α = rz1 / denom
            @. p += α*d
            @. r -= α*w
        end

        if norm2(r) ≤ ϵ_b
            return p
        else
            @. z = r / c
            rz2, rz1 = rz1, sum(r .* z)
        end
    end
    return p
end


"""
    npsd_precond_matrix(Ω₀, P)
"""
function npsd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
    r, s = size(Ω₀)

    c = ones(Float64, n)

    if r == 0 || r == n
        return c
    end

    H  = (P.^2)'
    H₁ = H[1:r,:]
    H₂ = H[(r+1):n,:]

    if r < s
        H12 = H₁' * Ω₀
        c  .= sum(H₁, dims=1)'.^2 .+ 2.0 * sum(H12 .* H₂', dims=2)
    else
        H12 = (1.0 .- Ω₀) * H₂
        c  .= sum(H, dims=1)'.^2 .- sum(H₂, dims=1)'.^2 .- 2.0 * sum(H₁ .* H12, dims=1)'
    end
    c[c .< 1e-8] .= 1e-8
    return c
end


"""
    npsd_set_omega(λ::Vector{Float64})
"""
function npsd_set_omega(λ::Vector{Float64}, n::Int)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        return zeros(Float64, 0, 0)
    elseif r == n
        return ones(Float64, n, n)
    else
        M  = Matrix{Float64}(undef, r, s)
        λᵣ = λ[1:r]
        λₛ = λ[(r+1):n]
        for j in 1:s, i in 1:r
            @inbounds M[i,j] = λᵣ[i] / (λᵣ[i] - λₛ[j])
        end
        return M
    end
end


"""
    npsd_jacobian(x, Ω₀, P; PERTURBATION=1e-9)
"""
function npsd_jacobian(x::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int; PERTURBATION::Float64=1e-9)
    r, s = size(Ω₀)

    if r == 0
        return zeros(Float64, n)
    elseif r == n
        return x .* (1.0 + PERTURBATION)
    end

    P₁ = P[:, 1:r]
    P₂ = P[:, (r+1):n]

    if r < s
        H₁ = diagm(x) * P₁
        Ω  = Ω₀ .* (H₁' * P₂)

        HT₁ = (P₁ * P₁') * H₁ .+ P₂ * Ω'
        HT₂ = P₁ * Ω

        return sum(P .* [HT₁ HT₂], dims=2) .+ x .* PERTURBATION
    else
        H₂ = diagm(x) * P₂
        Ω  = (1.0 .- Ω₀) .* (P₁' * H₂)

        HT₁ = P₂ * Ω'
        HT₂ = P₂ * H₂' * P₂ .+ P₁ * Ω

        return x .* (1.0 + PERTURBATION) .- sum(P .* [HT₁ HT₂], dims=2)
    end
end


"""
    cor_nearPSD(R::Matrix{Float64}{T};
        τ::Float64=1e-5,
        iter_outer::Int=200,
        iter_inner::Int=20,
        N::Int=200,
        err_tol::Float64=1e-6,
        precg_err_tol::Float64=1e-2,
        newton_err_tol::Float64=1e-4) where {T <: Float64}

Compute the nearest positive semidefinite correlation matrix given a symmetric
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

r = cor_nearPSD(ρ)
eigvals(r)
```
"""
function cor_nearPSD(R::Matrix{Float64};
    τ::Float64=1e-5,
    iter_outer::Int=200,
    iter_inner::Int=20,
    N::Int=200,
    err_tol::Float64=1e-6,
    precg_err_tol::Float64=1e-2,
    newton_err_tol::Float64=1e-4)

    n = size(R, 1)

    # Make R symmetric
    @. R = 0.5 * (R + R')

    b = ones(Float64, n)
    if τ > 0
        b .-= τ
        R .-= Matrix{Float64}(τ*I, n, n)
    end
    b₀ = copy(b)

    y    = zeros(Float64, n)
    X    = R .+ diagm(y)
    λ, P = eigen(X) # returns values in ascending order, but need descending
    λ   .= reverse(λ)
    P   .= reverse(P, dims=2)

    f₀, Fy = npsd_gradient(y, λ, P, b₀, n)
    f      = f₀
    b     .= b₀ .- Fy

    Ω₀ = npsd_set_omega(λ, n)
    x₀ = copy(y)

    X       .= npsd_pca(X, λ, P, n)
    val_R    = 0.5 * norm2(R)^2
    val_dual = val_R - f₀
    val_obj  = 0.5 * norm2(X - R)^2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

    normb  = norm2(b)
    normb0 = norm2(b₀) + 1
    Δnb    = normb / normb0

    k = 0
    c = Vector{Float64}(undef, n)
    d = Vector{Float64}(undef, n)
    while (gap > err_tol) && (Δnb > err_tol) && (k < iter_outer)
        c .= npsd_precond_matrix(Ω₀, P, n)
        d .= npsd_pre_cg(b, c, Ω₀, P, precg_err_tol, N, n)

        slope = sum((Fy .- b₀) .* d)

        y    .= x₀ .+ d
        X    .= R .+ diagm(y)
        λ, P  = eigen(X)
        λ    .= reverse(λ)
        P    .= reverse(P, dims=2)
        f, Fy = npsd_gradient(y, λ, P, b₀, n)

        k_inner = 0
        while (k_inner ≤ iter_inner) && (f > f₀ + newton_err_tol*slope*0.5^k_inner + 1e-6)
            k_inner += 1
            y    .= x₀ + d * 0.5^k_inner
            X    .= R .+ diagm(y)
            λ, P  = eigen(X)
            λ    .= reverse(λ)
            P    .= reverse(P, dims=2)
            f, Fy = npsd_gradient(y, λ, P, b₀, n)
        end

        x₀ .= y
        f₀  = f

        X       .= npsd_pca(X, λ, P, n)
        val_dual = val_R - f₀
        val_obj  = 0.5 * norm2(X - R)^2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b       .= b₀ .- Fy
        normb    = norm2(b)
        Δnb      = normb / normb0

        Ω₀ = npsd_set_omega(λ, n)

        k += 1
    end

    X .+= Matrix{Float64}(τ*I, n, n)
    cov2cor!(X)
    X
end

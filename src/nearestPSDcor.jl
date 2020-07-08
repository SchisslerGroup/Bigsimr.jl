function gradient(y, λ₀, P, b₀)
    n = length(λ₀)
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)

    if r == 0
        return (0, zeros(n))
    else
        λ[λ .< 0] .= 0
        Fy = begin
            # 1) M[:,j] = P[:,j] * λ[j]
            # 2) M = M .* M
            # 3) Fy = rowsums(M)
            _Fy = sum((P .* λ') .* P, dims=2)
            # 4) Fy = vec(Fy)
            vec(_Fy)
        end
        f  = 0.5 * sum(λ.^2) - sum(b₀ .* y)
        return (f, Fy)
    end
end


function PCA(X, λ, P)
    n = size(P, 1)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        return zeros(n, n)
    elseif r == 1
        return λ[1].^2 * P[:,1] * P[:,1]'
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
        return X + P₂λ₂ * P₂λ₂'
    end
end


function pre_cg(b, c, Ω₀, P, ϵ, N)
    n = size(P, 1)

    ϵ_b = ϵ * norm(b)

    r   = copy(b)
    z   = r ./ c
    rz1 = sum(r .* z)
    rz2 = 1.0

    p  = zeros(n)
    d  = copy(z)

    for k in 1:N
        if k > 1
            d .= z + d * (rz1 / rz2)
        end

        w = jacobian(d, Ω₀, P)

        denom = sum(d .* w)
        normr = norm(r)
        if denom ≤ 0
            return d / norm(d)
        else
            α = rz1 / denom
            p .+= α*d
            r .-= α*w
        end

        z .= r ./ c

        if norm(r) ≤ ϵ_b
            return p
        else
            rz2 = rz1
            rz1 = sum(r .* z)
        end
    end
    return p
end


function precond_matrix(Ω₀, P)
    n    = size(P, 1)
    r, s = size(Ω₀)

    c = ones(n)

    if r == 0 || r == n
        return c
    end

    H  = (P.^2)'
    H₁ = H[1:r,:]
    H₂ = H[(r+1):n,:]

    if r < s
        H12  = H₁' * Ω₀
        c   .= sum(H₁, dims=1).^2 + 2.0 * sum(H12 * H₂', dims=2)
    else
        H12  = (1.0 .- Ω₀) * H₂
        c   .= begin
                _a = vec(sum(H, dims=1).^2)            # column sums
                _b = vec(sum(H₂, dims=1).^2)           # column sums
                _c = vec(2.0 * sum(H₁ .* H12, dims=1)) # column sums
                _a - _b - _c
            end
    end
    c[c .< 1e-8] .= 1e-8
    return c
end


function set_omega(λ)
    n = length(λ)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        return zeros(0, 0)
    elseif r == n
        return ones(n, n)
    else
        M  = Array{Float64}(undef, r, s)
        λᵣ = λ[1:r]
        λₛ = λ[(r+1):n]
        for R in CartesianIndices(M)
            M[R] = λᵣ[R[1]] / (λᵣ[R[1]] - λₛ[R[2]])
        end
        return M
    end
end


function jacobian(x, Ω₀, P; PERTURBATION=1e-9)
    n    = size(P, 1)
    r, s = size(Ω₀)

    if 0 < r < n
        P₁ = P[:, 1:r]
        P₂ = P[:, (r+1):n]
    end

    if r == 0
        return zeros(n)
    elseif r == n
        return x * (1 + PERTURBATION)
    elseif r < s
        H₁ = diagm(x) * P₁
        Ω  = Ω₀ .* (H₁' * P₂)

        HT₁ = P₁ * P₁' * H₁ + P₂ * Ω'
        HT₂ = P₁ * Ω

        return vec(sum(P .* [HT₁ HT₂], dims=2)) + x * PERTURBATION
    else
        H₂ = diagm(x) * P₂
        Ω  = (1 .- Ω₀) .* (P₁' * H₂)

        HT₁ = P₂ * Ω'
        HT₂ = P₂ * H₂' * P₂ + P₁ * Ω

        return x * (1 + PERTURBATION) - vec(sum(P .* [HT₁ HT₂], dims=2))
    end
end


"""
    nearestPSDcor(R)

Compute the nearest positive semidefinite correlation matrix given a symmetric
correlation matrix `R`. This algorithm is based off of work by Qi and Sun 2006.
Matlab, C, R, and Python code can be found [on Sun's page](https://www.polyu.edu.hk/ama/profile/dfsun/index.html#Codes).
The algorithm has also been implemented in Fortran in the NAG library.

# Arguments
- `τ::Real`: a [small] nonnegative number used to enforce a minimum eigenvalue.
- `δ::Real`: the error tolerance for the stopping condition.

# Examples
```
import LinearAlgebra: eigvals
# Define a negative definite correlation matrix
ρ = [1.00 0.82 0.56 0.44
     0.82 1.00 0.28 0.85
     0.56 0.28 1.00 0.22
     0.44 0.85 0.22 1.00]
eigvals(ρ)

r = nearestPSDcor(ρ)
eigvals(r)
```
"""
function nearestPSDcor(R;
    τ::Real=1e-5,
    iter_outer=200,
    iter_inner=20,
    N=200,
    δ::Real=1e-6,  # error tol
    ϵ::Real=1e-2,  # pre-cg error tol
    σ::Real=1e-4)  # Newton method line search tol

    n = size(R, 1)

    # Make R symmetric
    R = (R + R') / 2

    b = ones(n)
    if τ > 0
        b = b .- τ
        R = R .- Array{eltype(R)}(τ*I, n, n)
    end
    b₀ = copy(b)

    y    = zeros(n)
    X    = R + diagm(y)
    λ, P = eigen(X) # returns values in ascending order, but need descending
    λ   .= reverse(λ)
    P   .= reverse(P, dims=2)

    f₀, Fy = gradient(y, λ, P, b₀)
    f      = f₀
    b     .= b₀ - Fy

    Ω₀ = set_omega(λ)
    x₀ = copy(y)

    X .= PCA(X, λ, P)
    val_R    = 0.5 * norm(R, 2)^2
    val_dual = val_R - f₀
    val_obj  = 0.5 * norm(X - R, 2)^2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

    normb  = norm(b)
    normb0 = norm(b₀) + 1
    Δnb    = normb / normb0

    k = 0
    while (gap > δ) && (Δnb > δ) && (k < iter_outer)
        c = precond_matrix(Ω₀, P)
        d = pre_cg(b, c, Ω₀, P, ϵ, N)

        slope = sum((Fy - b₀) .* d)

        y    .= x₀ + d
        X    .= R + diagm(y)
        λ, P  = eigen(X)
        λ    .= reverse(λ)
        P    .= reverse(P, dims=2)
        f, Fy = gradient(y, λ, P, b₀)

        k_inner = 0
        while (k_inner ≤ iter_inner) && (f > f₀ + σ*slope*0.5^k_inner + 1e-6)
            k_inner .+= 1
            y     .= x₀ + d * 0.5^k_inner
            X     .= R + diagm(y)
            λ, P  = eigen(X)
            λ    .= reverse(λ)
            P    .= reverse(P, dims=2)
            f, Fy = gradient(y, λ, P, b₀)
        end

        x₀ .= y
        f₀  = f

        X       .= PCA(X, λ, P)
        val_dual = val_R - f₀
        val_obj  = 0.5 * norm(X - R, 2)^2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b       .= b₀ - Fy
        normb    = norm(b)
        Δnb      = normb / normb0

        Ω₀ = set_omega(λ)

        k += 1
    end

    cov2cor(X .+ Array{eltype(R)}(τ*I, n, n))
end

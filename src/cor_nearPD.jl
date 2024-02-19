"""
    cor_nearPD(R::AbstractMatrix{<:Real}, τ=1e-6; tol=1e-3)

Return the nearest positive definite correlation matrix to `R`. `τ` is a
tuning parameter that controls the minimum eigenvalue of the resulting matrix.
`τ` can be set to zero if only a positive semidefinite matrix is needed. `tol`
is the accuracy in the relative gap. Set to a smaller value (e.g. `1e-8`) if
higher accuracy is needed.

See also: [`cor_fastPD`](@ref), [`cor_fastPD!`](@ref)

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

julia> p = cor_nearPD(r)
4×4 Matrix{Float64}:
 1.0       0.817095  0.559306  0.440514
 0.817095  1.0       0.280196  0.847352
 0.559306  0.280196  1.0       0.219582
 0.440514  0.847352  0.219582  1.0

julia> isposdef(p)
true
```
"""
function cor_nearPD(R::AbstractMatrix{T}, tau=1e-6; tol=1e-3) where {T<:Real}
    tau >= 0 || throw(ArgumentError("`tau` must be non-negative"))
    tol >= 0 || throw(ArgumentError("`tol` must be non-negative"))
    return _cor_nearPD(float(R), tau, tol)
end

function _cor_nearPD(R::AbstractMatrix{T}, tau, tol) where {T<:Real}
    # Setup
    n = size(R, 1)
    iter_outer = 200
    iter_inner = 20
    iter_cg    = 200
    tol_cg     = 1e-2
    tol_ls     = 1e-4
    err_tol    = max(eps(T), tol)

    # Make R symmetric
    R .= Symmetric(R, :U) # [n,n]
    R[diagind(R)] .= one(Float64)

    b = ones(T, n)
    b .-= tau
    R[diagind(R)] .-= T(tau)
    b₀ = copy(b)

    y    = zeros(T, n)        # [n,1]
    X    = copy(R)            # [n,n]
    λ, P = eigen(X)           # [n,1], [n,n]
    λ, P = T.(λ), T.(P)
    λ   .= reverse(λ)         # [n,1]
    P   .= reverse(P, dims=2) # [n,n]

    f₀, Fy = _npd_gradient(y, λ, P, b₀, n) # [1], [n,1]
    f      = f₀      # [1]
    b     .= b₀ - Fy # [n,1]

    Ω₀ = _npd_set_omega(λ, n) # [r,s]
    x₀ = copy(y)              # [n,1]

    X       .= _npd_pca(b₀, X, λ, P, n) # [n,n]
    val_R    = 0.5 * norm(R)^2
    val_dual = val_R - f₀
    val_obj  = 0.5 * norm(X - R)^2
    gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))

    norm_b  = norm(b)
    norm_b0 = norm(b₀) + 1
    norm_b_rel = norm_b / norm_b0

    k = 0
    while (gap > err_tol) && (norm_b_rel > err_tol) && (k < iter_outer)
        c = _npd_precond_matrix(Ω₀, P, n)                # [n,1]
        d = _npd_pre_cg(b, c, Ω₀, P, tol_cg, iter_cg, n) # [n,1]

        slope = sum((Fy - b₀) .* d)          # [1]
        y    .= x₀ + d                       # [n,1]
        X    .= R + diagm(y)                 # [n,n]
        λ, P  = eigen(X)                     # [n,1], [n,n]
        λ, P = T.(λ), T.(P)
        λ    .= reverse(λ)                   # [n,1]
        P    .= reverse(P, dims=2)           # [n,n]
        f, Fy = _npd_gradient(y, λ, P, b₀, n) # [1], [n,1]

        k_inner = 0
        while (k_inner ≤ iter_inner) && (f > f₀ + tol_ls * slope * 0.5^k_inner + 1e-6)
            k_inner += 1
            y    .= x₀ + d * 0.5^k_inner         # [n,1]
            X    .= R + diagm(y)                 # [n,n]
            λ, P  = eigen(X)                     # [n,1], [n,n]
            λ, P = T.(λ), T.(P)
            λ    .= reverse(λ)                   # [n,1], [n,n]
            P    .= reverse(P, dims=2)           # [n,n]
            f, Fy = _npd_gradient(y, λ, P, b₀, n) # [1], [n,1]
        end

        x₀  = copy(y) # [n,1]
        f₀  = f

        X       .= _npd_pca(b₀, X, λ, P, n) # [n,n]
        val_dual = val_R - f₀
        val_obj  = 0.5 * norm(X - R)^2
        gap      = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        b        = b₀ - Fy
        norm_b   = norm(b)
        norm_b_rel      = norm_b / norm_b0

        Ω₀ = _npd_set_omega(λ, n) # [n,n] or [r,s]

        k += 1
    end

    X[diagind(X)] .+= T(tau)
    return cov2cor!(X)
end



#=
    Return f(yₖ) and ∇f(yₖ) where

    ```math
    f(y) = \\frac{1}{2} \\Vert (A + diag(y))_+ \\Vert_{F}^{2} - e^{T}y
    ```

    and

    ```math
    \\nabla f(y) = Diag((A + diag(y))_+) - e
    ```
=#
function _npd_gradient(y, λ₀, P, b₀, n)
    T = eltype(y)
    r = sum(λ₀ .> 0)
    λ = copy(λ₀)

    r == 0 && return (zero(T), zeros(T, n))

    λ[λ .< 0] .= zero(T)
    Fy = Vector{T}(vec(sum((P .* λ') .* P, dims=2)))
    f  = T(T(0.5) * sum(λ.^2) - sum(b₀ .* y))

    return (f, Fy)
end



function _npd_pca(b, X, λ, P, n)
    T = eltype(b)
    r = sum(λ .> 0)
    s = n - r

    if r == 0
        X .= zeros(T, n, n)
    elseif r == n
        nothing
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

    d  = diag(X)
    d .= max.(d, b)
    X[diagind(X)] .= d
    d .= sqrt.(b ./ d)
    d₂ = d * d'
    X .= X .* d₂

    return X
end



# Preconditioned conjugate gradient method to solve Vₖdₖ = -∇f(yₖ)
function _npd_pre_cg(b, c, Ω_0, P, ϵ, N, n)
    T   = eltype(b)
    ϵ_b = T(ϵ) * norm(b)
    r   = copy(b)
    z   = r ./ c
    d   = copy(z)
    rz1 = sum(r .* z)
    rz2 = one(T)
    p   = zeros(T, n)
    w   = zeros(T, n)

    for k in 1:N
        if k > 1
            d .= z + d * (rz1 / rz2)
        end

        w .= _npd_jacobian(d, Ω_0, P, n)

        denom = sum(d .* w)

        denom ≤ 0 && return T.(d / norm(d))

        a = rz1 / denom
        p .+= a*d
        r .-= a*w

        norm(r) ≤ ϵ_b && return T.(p)

        z .= r ./ c
        rz2, rz1 = copy(rz1), sum(r .* z)
    end

    T.(p)
end



# Create the precondition matrix used in solving the linear system
# Vₖdₖ = -∇f(yₖ) in the conjugate gradient method.
function _npd_precond_matrix(W, P, n)
    T = eltype(W)
    r, s = size(W)

    r == 0 || r == n && return ones(T, n)

    H  = adjoint(P .* P)
    H₁ = @view H[1:r,:]
    H₂ = @view H[r+1:n,:]

    if r < s
        H12 = H₁' * W
        c   = sum(H₁, dims=1)'.^2 + 2 * sum(H12 .* H₂', dims=2)
    else
        H12 = (1.0 .- W) * H₂
        c   = sum(H, dims=1)'.^2 - sum(H₂, dims=1)'.^2 - 2 * sum(H₁ .* H12, dims=1)'
    end

    c[c .< 1e-8] .= T(1e-8)
    return T.(c)
end



# Used in creating the precondition matrix.
function _npd_set_omega(λ, n)
    T = eltype(λ)
    r = sum(λ .> 0)
    s = n - r

    r == 0 && return zeros(T, 0, 0)
    r == n && return ones(T, n, n)

    M = zeros(T, r, s)
    λr = @view λ[1:r]
    λs = @view λ[r+1:n]

    @inbounds for j = 1:s, i = 1:r
        M[i,j] = λr[i] / (λr[i] - λs[j])
    end

    T.(M)
end



# Create the Generalized Jacobian matrix for the Newton direction step.
function _npd_jacobian(x, W0, P, n)
    T = eltype(x)
    r, s = size(W0)
    perturbation = 1e-10

    r == 0 && return zeros(T, n)
    r == n && return T.(x .* (1 + perturbation))

    P₁ = @view P[:, 1:r]
    P₂ = @view P[:, r+1:n]

    if r < s
        H1 = diagm(x) * P₁
        W  = W0 .* (H1' * P₂)

        HT1 = P₁ * P₁' * H1 + P₂ * W'
        HT2 = P₁ * W

        return T.(vec(sum(P .* [HT1 HT2], dims=2) + x .* perturbation))
    else
        H₂ = diagm(x) * P₂
        W  = (1 .- W0) .* (P₁' * H₂)

        HT1 = P₂ * W'
        HT2 = P₂ * H₂' * P₂ + P₁ * W

        return T.(vec(x .* (1 + perturbation) - sum(P .* [HT1 HT2], dims=2)))
    end
end

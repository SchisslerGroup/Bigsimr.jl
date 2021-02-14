function get_coefs(margin::UD, n::Int)
    aₖ = zeros(Float64, n + 1)
    m = 2n
    tₛ, wₛ = gausshermite(m)
    tₛ    .= tₛ * sqrt2
    Xₛ = normal_to_margin(margin, tₛ)

    aₖ = [sum(wₛ .* _h(tₛ, k) .* Xₛ) for k in 0:n]
    
    return invsqrtpi * aₖ ./ factorial.(0:n)
end
get_coefs(margins::UD, n::Real) = get_coefs(margins, Int(n))

function Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)
    if n == 0
        return 0.0
    end
    M = length(A)
    N = length(B)
    accu = 0.0
    for r=1:M, s=1:N
        r11 = Hp(α[r+1], n-1) * Hp(β[s+1], n-1)
        r00 = Hp(α[r],   n-1) * Hp(β[s],   n-1)
        r01 = Hp(α[r],   n-1) * Hp(β[s+1], n-1)
        r10 = Hp(α[r+1], n-1) * Hp(β[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end
    accu * σAσB_inv
end

function Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UD, σAσB_inv::Float64)
    if n == 0
        return 0.0
    end
    M = length(A)
    accu = 0.0
    for r=1:M
        accu += A[r] * (Hp(α[r+1], n-1) - Hp(α[r], n-1))
    end
    m = n + 4
    t, w = gausshermite(m)
    t .= t * sqrt2
    X = normal_to_margin(dB, t)
    S = invsqrtpi * sum(w .* _h(t, n) .* X)
    return -σAσB_inv * accu * S
end

function solve_poly_pm_one(coef::Vector{Float64})
    P = Polynomial(coef)
	dP = derivative(P)
    r = roots(x->P(x), x->dP(x), -1..1, Krawczyk, 1e-3)
        
    length(r) == 1 && return mid(r[1].interval)
    length(r) == 0 && return NaN
    length(r) > 1 && error("More than one root found in the interval -1..1")
end


function _h(x::T, n::Int) where {T<:Real}
    if n == 0
        return one(T)
    elseif n == 1
        return x
    end
    
    Hkp1, Hk, Hkm1 = zero(T), x, one(T)
    for k in 2:n
        Hkp1 = x*Hk - (k-1) * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end
    Hkp1
end
_h(X::Real, n::Real) = _h(X, Int(n))
_h(A::Array{<:Real, N}, n::Real) where N = _h.(A, Ref(n))

# We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
function Hp(x::Real, n::Int)
    isinf(x) ? zero(x) : _h(x, n) * _normpdf(x)
end
Hp(x::Real, n::Real) = Hp(x, Int(n))
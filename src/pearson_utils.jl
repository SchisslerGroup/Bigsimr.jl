function _get_coefs(margin, n::Int)
    m = 2n
    t, w = gausshermite(m)
    t .*= sqrt2
    X = _norm2margin(margin, t)

    ak = [sum(w .* _h(t, k) .* X) for k in 0:n]

    return invsqrtπ * ak ./ factorial.(0:n)
end



function _Gn0d(n::Int, A, B, xs, ys, cov_inv)
    n == 0 && return zero(Float64)
    M = length(A)
    N = length(B)

    accu = zero(Float64)

    for r=1:M, s=1:N
        r11 = _Hp(xs[r+1], n-1) * _Hp(ys[s+1], n-1)
        r00 = _Hp(xs[r],   n-1) * _Hp(ys[s],   n-1)
        r01 = _Hp(xs[r],   n-1) * _Hp(ys[s+1], n-1)
        r10 = _Hp(xs[r+1], n-1) * _Hp(ys[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end

    return accu * cov_inv
end



function _Gn0m(n::Int, A, xs, dB, cov_inv)
    n == 0 && return zero(Float64)
    M = length(A)

    accu = zero(Float64)

    for r = 1:M
        accu += A[r] * (_Hp(xs[r+1], n-1) - _Hp(xs[r], n-1))
    end

    m = n + 4
    t, w = gausshermite(m)
    t .*= sqrt2
    X = _norm2margin(dB, t)
    S = invsqrtπ * sum(w .* _h(t, n) .* X)

    return -cov_inv * accu * S
end


3
function _find_roots_pm1(coef)
    P = Polynomial(coef)
	dP = derivative(P)
    rs = roots(x -> P(x), x -> dP(x), interval(-1, 1), Krawczyk, 1e-3)
    return map(r -> mid(interval(r)), rs)
end

# returns the only root, or the root nearest to the target, otherwise NaN
function _find_root_or_nan(coef, target)
    rs = _find_roots_pm1(coef)
    nr = length(rs)

    nr == 1 && return only(rs)
    nr == 0 && return NaN

    # multiple roots, return the one nearest to the target
    return _nearest_root(target, rs)
end

# returns the only root or NaN
function _find_root_or_nan(coef)
    rs = _find_roots_pm1(coef)
    return length(rs) == 1 ? only(rs) : NaN
end

# finds the root nearest to the target
function _nearest_root(target, roots)
    i = argmin(abs(r - target) for r in  roots)
    return roots[i]
end



function _h(x::Real, n::Int)
    n == 0 && return one(x)
    n == 1 && return x

    Hkp1, Hk, Hkm1 = zero(x), x, one(x)

    for k in 2:n
        Hkp1 = x*Hk - (k-1) * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end

    return Hkp1
end

_h(A::AbstractArray{<:Real,N}, n::Int) where {N} = _h.(A, Ref(n))



# We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
_Hp(x, n) = isinf(x) ? zero(x) : _h(x, n) * _normpdf(x)

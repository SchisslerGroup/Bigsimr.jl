function _get_coefs(margin, n)
    m = 2n
    t, w = gausshermite(m)
    t .*= sqrt2_f64
    X = _norm2margin(margin, t)

    ak = [sum(w .* _h(t, k) .* X) for k in 0:n]
    
    return invsqrtpi_f64 * ak ./ factorial.(0:n)
end



function _Gn0d(n, A, B, xs, ys, cov_inv)
    n == 0 && return 0.0
    M = length(A)
    N = length(B)

    accu = 0

    for r=1:M, s=1:N
        r11 = _Hp(xs[r+1], n-1) * _Hp(ys[s+1], n-1)
        r00 = _Hp(xs[r],   n-1) * _Hp(ys[s],   n-1)
        r01 = _Hp(xs[r],   n-1) * _Hp(ys[s+1], n-1)
        r10 = _Hp(xs[r+1], n-1) * _Hp(ys[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end

    return accu * cov_inv
end



function _Gn0m(n, A, xs, dB, cov_inv)
    n == 0 && return 0
    M = length(A)
    
    accu = 0
    
    for r = 1:M
        accu += A[r] * (_Hp(xs[r+1], n-1) - _Hp(xs[r], n-1))
    end
    
    m = n + 4
    t, w = gausshermite(m)
    t .*= sqrt2_f64
    X = _norm2margin(dB, t)
    S = invsqrtpi_f64 * sum(w .* _h(t, n) .* X)

    return -cov_inv * accu * S
end



function _solve_poly_pm_one(coef)
    P = Polynomial(coef)
	dP = derivative(P)
    r = roots(x -> P(x), x -> dP(x), interval(-1, 1), Krawczyk, 1e-3)

    nr = length(r)

    nr == 1 && return mid(first(r).interval)
    nr == 0 && return NaN
    
    return [mid(rs.interval) for rs in r]
end



_nearest_root(target, roots) = roots[argmin(abs.(roots .- target))]



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

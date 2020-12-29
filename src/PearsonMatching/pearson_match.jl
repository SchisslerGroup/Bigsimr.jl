"""
    pearson_match(ρ::Real, dA::UD, dB::UD; n::Int=7)

Compute the pearson correlation coefficient that is necessary to achieve the
target correlation given a pair of marginal distributions.
"""
function pearson_match(ρ::Real, dA::UD, dB::UD; n::Int=7)
    _pearson_match(ρ, dA, dB, n)
end


function _pearson_match(ρ::Real, dA::CUD, dB::CUD, n::Int)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)

    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = c2 .* a[k+1] * b[k+1] * factorial(k)
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - ρ

    r = solve_poly_pm_one(coef)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    ρ_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρ_u = c1 * c2 + c2 * sum(kab)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end

function _pearson_match(ρ::Real, dA::DUD, dB::DUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    minB = minimum(dB)
    maxA = maximum(dA)
    maxB = maximum(dB)

    maxA = isinf(maxA) ? quantile(dA, 0.99999) : maxA
    maxB = isinf(maxB) ? quantile(dB, 0.99999) : maxB

    TA = eltype(dA)
    TB = eltype(dB)

    # Support sets
    A = range(minA, maxA, step=1.0)
    B = range(minB, maxB, step=1.0)

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]
    β = [-Inf; quantile.(Normal(), cdf.(dB, B))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = Gn0d(k, A, B, α, β, c2) / factorial(k)
    end
    coef[1] = -ρ

    r = solve_poly_pm_one(coef)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    ρ_l, ρ_u = pearson_bounds(dA, dB)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end

function _pearson_match(ρ::Real, dA::DUD, dB::CUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    maxA = maximum(dA)

    maxA = isinf(maxA) ? quantile(dA, 0.99999) : maxA

    TA = eltype(dA)
    A = range(minA, maxA, step=1.0)
    α = [-Inf; quantile.(Normal(), cdf.(dA, A))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = Gn0m(k, A, α, dB, c2) / factorial(k)
    end
    coef[1] = -ρ

    r = solve_poly_pm_one(coef)
    !isnan(r) && return r

    #= 
        If the root does not exist, then compute the adjustment correlation for
        the theoretical upper or lower correlation bound.
    =#
    ρ_l, ρ_u = pearson_bounds(dA, dB)
    ρ > 0 ? _pearson_match(ρ_u-0.001, dA, dB, n) : _pearson_match(ρ_l+0.001, dA, dB, n)
end

_pearson_match(ρ::Real, dA::CUD, dB::DUD, n::Int) = _pearson_match(ρ, dB, dA, n)


"""
    pearson_match(D::MvDistribution; n::Int=7)
"""
function pearson_match(D::MvDistribution; n::Int=7)
    d = length(D.F)

    # Make sure that ρ is a Pearson correlation
    R = cor_convert(cor(D), cortype(D), Pearson)

    # Calculate the pearson matching pairs
    @threads for i in collect(subsets(1:d, Val{2}()))
        @inbounds R[i...] = pearson_match(D.ρ[i...], D.F[i[1]], D.F[i[2]], n=n)
    end

    # Ensure that the resulting correlation matrix is positive definite
    R .= cor_nearPD(Matrix{eltype(D)}(Symmetric(R)))

    # Return the new MvDistribution
    MvDistribution(R, margins(D), Pearson)
end
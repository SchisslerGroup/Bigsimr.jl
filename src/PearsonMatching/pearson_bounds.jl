"""
    pearson_bounds

Compute the lower and upper bounds of possible correlations for a pair of
univariate distributions. The value `n` determines the accuracy of the 
approximation of the two distributions.
"""
function pearson_bounds end

function pearson_bounds(dA::UD, dB::UD, μA, μB, σA, σB; n::Integer=7)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρ_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρ_u = c1 * c2 + c2 * sum(kab)

    ρ_l, ρ_u = clampcor.((ρ_l, ρ_u))
    (lower = ρ_l, upper = ρ_u)
end


function pearson_bounds(dA::UD, dB::UD)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)
    pearson_bounds(dA, dB, μA, μB, σA, σB)
end


function pearson_bounds(D::MvDistribution)
    d = length(D.F)

    lower, upper = similar(cor(D)), similar(cor(D))

    @threads for i in collect(subsets(1:d, Val{2}()))
        l, u = pearson_bounds(D.F[i[1]], D.F[i[2]])
        lower[i...] = l
        upper[i...] = u
    end

    lower .= cor_constrain(Matrix{eltype(D)}(Symmetric(lower)))
    upper .= cor_constrain(Matrix{eltype(D)}(Symmetric(upper)))

    (lower = lower, upper = upper)
end
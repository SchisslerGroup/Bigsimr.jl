"""
    ρz_bounds

Compute the lower and upper bounds of possible correlations for a pair of
univariate distributions. The value `n` determines the accuracy of the 
approximation of the two distributions.
"""
function ρz_bounds end

function ρz_bounds(dA::UD, dB::UD, μA, μB, σA, σB; n::Integer=7)
    k = 0:1:n
    a = get_coefs(dA, n)
    b = get_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρx_l = c1 * c2 + c2 * sum((-1) .^ k .* kab)
    ρx_u = c1 * c2 + c2 * sum(kab)

    clampcor.((ρx_l, ρx_u))
end


function ρz_bounds(dA::UD, dB::UD)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)
    ρz_bounds(dA, dB, μA, μB, σA, σB)
end
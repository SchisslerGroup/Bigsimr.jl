function cov2cor(rho::AbstractArray)
    D = pinv(diagm(sqrt.(diag(rho))))
    D .= D * rho * D
    diag(D) .= 1.0
    (D + D') / 2
end


function rcor(d::Integer, k::Integer=1)
    W = randn(Float64, d, k)
    S = W * W' + diagm(rand(Float64, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    S2 .= S2 * S * S2
    diag(S2) .= 1.0
    S2
end

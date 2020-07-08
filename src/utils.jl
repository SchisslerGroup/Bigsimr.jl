"""
    cov2cor(rho::AbstractArray)

Convert a covariance matrix to a correlation matrix. Ensure that the resulting
matrix is symmetric and has diagonals equal to 1.0.
"""
function cov2cor(rho::AbstractArray)
    D = pinv(diagm(sqrt.(diag(rho))))
    D .= D * rho * D
    diag(D) .= 1.0
    (D + D') / 2
end


"""
    rcor(d::Integer, k::Integer=1)

Generate a random correlation matrix of size ``d×d``. the parameter `k` is used
to set the factor loadings for ``W``. The code has been adapted from user *amoeba*
from [StackExchange](https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor)
"""
function rcor(d::Integer, k::Integer=1)
    W = randn(Float64, d, k)
    S = W * W' + diagm(rand(Float64, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    S2 .= S2 * S * S2
    diag(S2) .= 1.0
    S2
end

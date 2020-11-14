using MvSim
using Test
import LinearAlgebra: eigvals, diag, isposdef

ρ = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]

ρ_hat = cor_nearPSD(ρ, n_iter=100)
λ = eigvals(ρ_hat)
@test all(λ .≥ 0)
@test all(diag(ρ_hat) .== 1.0)
@test ρ_hat ≈ ρ_hat' atol=1e-12
@test all(-1.0 .≤ ρ_hat .≤ 1.0)
isposdef(ρ_hat)

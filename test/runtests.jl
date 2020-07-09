using MvSim
using Test

import LinearAlgebra: eigvals, diag

@testset "Utilities" begin
    @test_throws ArgumentError cor2cor(0.0, "from_not_valid", "pearson")

    S = [0.99 0.78 0.59 0.44
         0.78 0.92 0.28 0.81
         0.59 0.28 1.12 0.23
         0.44 0.81 0.23 0.99]

    # Test that covariance to correlation conversion is working
    ρ = cov2cor(S)
    @test all(diag(ρ) .== 1.0)
    @test ρ == ρ'
    @test all(-1.0 .≤ ρ .≤ 1.0)

    # Hermite polynomials
    He5(x) = x.^5 .- 10x.^3 .+ 15x
    H5(x) = 32x.^5 .- 160x.^3 .+ 120x
    x = -10:0.1:10
    @test all(hermite(x, 5) .≈ He5(x))
    @test all(hermite(x, 5, false) .≈ H5(x))

    # Test that random correlation generation is working
    r = rcor(10)
    @test all(diag(r) .== 1.0)
    @test r ≈ r' atol=1e-12
    @test all(-1.0 .≤ r .≤ 1.0)
    λ = eigvals(r)
    @test all(λ .≥ 0)
end


@testset "Nearest PSD correlation" begin
    ρ = [1.00 0.82 0.56 0.44
         0.82 1.00 0.28 0.85
         0.56 0.28 1.00 0.22
         0.44 0.85 0.22 1.00]

    # Test that it returns the nearest positive semidefinite correlation matrix
    ρ_hat = nearestPSDcor(ρ)
    λ = eigvals(ρ_hat)
    @test all(λ .≥ 0)
    @test all(diag(ρ_hat) .== 1.0)
    @test ρ_hat ≈ ρ_hat' atol=1e-12
    @test all(-1.0 .≤ ρ_hat .≤ 1.0)
end

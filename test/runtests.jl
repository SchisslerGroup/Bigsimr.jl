using MvSim
using Test

import LinearAlgebra: eigvals, diag, isposdef

@testset "MvSim.jl" begin
    S = [0.99 0.78 0.59 0.44
         0.78 0.92 0.28 0.81
         0.59 0.28 1.12 0.23
         0.44 0.81 0.23 0.99]

    # Test that covariance to correlation conversion is working
    ρ = cov2cor(S)
    @test all(diag(ρ) .≈ 1.0)
    @test ρ == ρ'
    @test all(prevfloat(-1.0) .≤ ρ .≤ nextfloat(1.0))

    # Test that random correlation generation is working
    r = rcor(10)
    @test all(diag(r) .≈ 1.0)
    @test r ≈ r'
    @test all(prevfloat(-1.0) .≤ r .≤ nextfloat(1.0))
    λ = eigvals(r)
    @test all(λ .≥ 0)

    # Test that it returns the nearest positive semidefinite correlation matrix
    ρhat = nearestSPDcor(ρ)
    λ = eigvals(ρhat)
    @test all(λ .≥ 0)
    @test all(diag(ρhat) .≈ 1.0)
    @test ρhat == ρhat'
    @test all(prevfloat(-1.0) .≤ ρhat .≤ nextfloat(1.0))
end

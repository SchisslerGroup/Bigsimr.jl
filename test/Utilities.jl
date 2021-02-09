using Test
using bigsimr
using Distributions

@testset "Utilities" begin

    @testset "Hermite Polynomials" begin
        He5(x) = x.^5 .- 10x.^3 .+ 15x     # Known Probabilists 5th degree
        H5(x) = 32x.^5 .- 160x.^3 .+ 120x  # Known Physicists 5th degree
        x = 200 * rand(100) .- 100
        @test all(bigsimr.hermite.(x, 5) .≈ He5(x))
        @test all(bigsimr.hermite.(x, 5, false) .≈ H5(x))
    end

    @testset "Normal to Marginal" begin
        # Must work for scalars, vectors, and matrices of type Float64
        D = Binomial(10, 0.5)
        
        x = randn(Float64)
        y = randn(Float64, 2)
        z = randn(Float64, 2, 2)

        ω = randn(Float64, 2, 2, 2)
        γ = randn(Float32)

        @test_nowarn bigsimr.normal_to_margin(D, x)
        @test_nowarn bigsimr.normal_to_margin(D, y)
        @test_nowarn bigsimr.normal_to_margin(D, z)

        @test_throws MethodError bigsimr.normal_to_margin(D, ω)
        @test_throws MethodError bigsimr.normal_to_margin(D, γ)

        # Standard normal to standard normal should be invariant
        z = rand(Normal(0, 1), 100000)
        @test z ≈ bigsimr.normal_to_margin(Normal(0, 1), z)

        # Estimated parameters must be close to true parameters
        d1 = Binomial(20, 0.2)
        d2 = Poisson(3)
        d3 = Normal(12, π)

        x1 = bigsimr.normal_to_margin(d1, z)
        x2 = bigsimr.normal_to_margin(d2, z)
        x3 = bigsimr.normal_to_margin(d3, z)

        f1 = fit_mle(Binomial, 20, x1)
        f2 = fit_mle(Poisson, x2)
        f3 = fit_mle(Normal, x3)

        @test all(isapprox.(params(d1), params(f1), rtol=0.01))
        @test all(isapprox.(params(d2), params(f2), rtol=0.01))
        @test all(isapprox.(params(d3), params(f3), rtol=0.01))
    end

end
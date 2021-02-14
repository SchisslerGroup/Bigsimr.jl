using Test
using Bigsimr
using Distributions
import LinearAlgebra: PosDefException
import Bigsimr: ValidCorrelationError

@testset "Random Vector Utilities" begin

    @testset "Fast Standard Random Normal" begin
        # Must work for each bits floating point type
        for T in (Float64, Float32, Float16)
            @test_nowarn Bigsimr._randn(T, 4, 4)
            @test eltype(Bigsimr._randn(T, 4, 4)) === T
        end

        @test_nowarn Bigsimr._randn(Float64, 4.0, 4.0)
        @test_nowarn Bigsimr._randn(Float64, 4.0, 4)
        @test_nowarn Bigsimr._randn(Float64, 4,   4.0)

        @test_nowarn Bigsimr._randn(4.0, 4.0)
        @test_nowarn Bigsimr._randn(4.0, 4)
        @test_nowarn Bigsimr._randn(4  , 4.0)

        @test_throws InexactError Bigsimr._randn(4.5, 5.5)
        @test_throws InexactError Bigsimr._randn(4.5, 5)
        @test_throws InexactError Bigsimr._randn(4  , 5.5)
    end

    @testset "Random Multivariate Normal Generation" begin
        r_negdef = [
            1.00 0.82 0.56 0.44
            0.82 1.00 0.28 0.85
            0.56 0.28 1.00 0.22
            0.44 0.85 0.22 1.00
        ]

        # Must fail for negative semidefinite matrices
        @test_throws PosDefException Bigsimr._rmvn(10, r_negdef)

        # Must work for each bits floating point type
        for T in (Float64, Float32, Float16)
            r = cor_randPD(T, 4)
            @test_nowarn Bigsimr._rmvn(10, r)
            @test eltype(Bigsimr._rmvn(10, r)) === T
        end
    end

    @testset "User Random Multivariate Normal Generation" begin
        # Must work for each bits floating point type
        for T in (Float64, Float32, Float16)
            r = cor_randPD(T, 4)
            @test_nowarn rmvn(10, r)
            @test eltype(rmvn(10, r)) === T
        end

        r = cor_randPD(4)
        @test_nowarn rmvn(10.0, r)
        @test_throws InexactError rmvn(10.5, r)
    end

    @testset "Normal to Marginal" begin
        # Must work for scalars, vectors, and matrices of type Float64
        D = Binomial(10, 0.5)
        
        x = randn(Float64)
        y = randn(Float64, 2)
        z = randn(Float64, 2, 2)

        ω = randn(Float64, 2, 2, 2)
        γ = randn(Float32)

        @test_nowarn Bigsimr.normal_to_margin(D, x)
        @test_nowarn Bigsimr.normal_to_margin(D, y)
        @test_nowarn Bigsimr.normal_to_margin(D, z)

        @test_nowarn Bigsimr.normal_to_margin(D, ω)
        @test_nowarn Bigsimr.normal_to_margin(D, γ)

        # Standard normal to standard normal should be invariant
        z = rand(Normal(0, 1), 100000)
        @test z ≈ Bigsimr.normal_to_margin(Normal(0, 1), z)

        # Estimated parameters must be close to true parameters
        d1 = Binomial(20, 0.2)
        d2 = Poisson(3)
        d3 = Normal(12, π)

        x1 = Bigsimr.normal_to_margin(d1, z)
        x2 = Bigsimr.normal_to_margin(d2, z)
        x3 = Bigsimr.normal_to_margin(d3, z)

        f1 = fit_mle(Binomial, 20, x1)
        f2 = fit_mle(Poisson, x2)
        f3 = fit_mle(Normal, x3)

        @test all(isapprox.(params(d1), params(f1), rtol=0.01))
        @test all(isapprox.(params(d2), params(f2), rtol=0.01))
        @test all(isapprox.(params(d3), params(f3), rtol=0.01))
    end

end

@testset "Random Vector Generation" begin

    @testset "rvec" begin
        # Must throw an error if a margin is not a univariate distribution
        r = cor_randPD(2)
        m = [Binomial(10, 0.2), MvNormal(zeros(2), r)]
        @test_throws MethodError rvec(2, r, m)

        # Must throw an error if r is not a valid correlation matrix
        m = [Binomial(10, 0.3), Gamma(10, 3)]
        r = Float64[1.0 2.33333; 0.333333 1.0] # Not positive definite
        c = Float64[2 4; 4 100]                # Is covariance, not correlation
        @test_throws ValidCorrelationError rvec(3, r, m)
        @test_throws ValidCorrelationError rvec(3, c, m)

        # Must throw an arror if the dimensions of r do not match the number of margins
        m = [Binomial(10, 0.3), Gamma(10, 3)]
        r = cor_randPD(3)
        @test_throws DimensionMismatch rvec(4, r, m)

        # Distributions.jl only returns random Float64 types, but `rvec` should
        # still accept correlation matrices of different floating point types
        for T in (Float64, Float32, Float16)
            r = cor_randPD(T, 2)
            @test_nowarn rvec(10, r, m)
            @test_nowarn rvec(10.0, r, m)
            @test_throws InexactError rvec(10.5, r, m)
        end
    end

end
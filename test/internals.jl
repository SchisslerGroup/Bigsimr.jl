using Test, Bigsimr
using Distributions

import Bigsimr.BigsimrBase as Utils

include("test_macros.jl")

@testset "Internal Utilities" begin
    @testset "Normal to Marginal" begin
        D = Binomial(10, 0.5)

        # Must work for scalars, vectors, and matrices of type Float64
        @test_nothrow Utils.norm2margin(D, randn())
        @test_nothrow Utils.norm2margin(D, randn(2))
        @test_nothrow Utils.norm2margin(D, randn(2, 2))
        @test_nothrow Utils.norm2margin(D, randn(2, 2, 2))

        # Must work for other input types
        @test_nothrow Utils.norm2margin(D, 1)
        @test_nothrow Utils.norm2margin(D, 1//2)
        @test_nothrow Utils.norm2margin(D, π)
        @test_nothrow Utils.norm2margin(D, 0.57)
        @test_nothrow Utils.norm2margin(D, 0.57f0)
        @test_nothrow Utils.norm2margin(D, Float16(0.57))
        @test_nothrow Utils.norm2margin(D, BigFloat(0.57))

        # Standard normal to standard normal should be invariant
        z = rand(Normal(0, 1), 100_000)
        @test z ≈ Utils.norm2margin(Normal(0, 1), z)

        # Estimated parameters must be close to true parameters
        d1 = Binomial(20, 0.2)
        d2 = Poisson(3)
        d3 = Normal(12, π)

        x1 = Utils.norm2margin(d1, z)
        x2 = Utils.norm2margin(d2, z)
        x3 = Utils.norm2margin(d3, z)

        f1 = fit_mle(Binomial, 20, x1)
        f2 = fit_mle(Poisson, x2)
        f3 = fit_mle(Normal, x3)

        @test all(isapprox.(params(d1), params(f1), rtol=0.01))
        @test all(isapprox.(params(d2), params(f2), rtol=0.01))
        @test all(isapprox.(params(d3), params(f3), rtol=0.01))
    end

    @testset "Random Normal" begin
        # General usage
        @test_nothrow Utils.randn_shared(Float64, 100, 10)
        @test_nothrow Utils.randn_shared(100, 10)

        # Must work for common floating-point types
        for T in (Float16, Float32, Float64)
            Z = Utils.randn_shared(T, 100_000, 10)
            @test eltype(Z) === T
        end
    end

    @testset "Random Multivariate Normal" begin
        r = Utils.make_negdef_matrix(Float64)
        rho = cor_nearPD(r)

        # General usage
        @test_nothrow Utils.rmvn_shared(100, rho)
        @test_nothrow Utils.rmvn_shared(100, 0.5)
    end

    @testset "Other Helpers" begin
        X = reshape(collect(1:16), 4, 4)

        @test_nothrow Utils.idx_subsets2(10)
        @test_nothrow Utils.make_symmetric!(X)
        @test_nothrow Utils.set_diag1!(X)

        @test_nothrow Utils.clampcor(BigFloat(3))
        @test_nothrow Utils.clampcor(3.14)
        @test_nothrow Utils.clampcor(3.14f0)
        @test_nothrow Utils.clampcor(Float16(3))
        @test_nothrow Utils.clampcor(BigInt(3))
        @test_nothrow Utils.clampcor(3)
        @test_nothrow Utils.clampcor(3//1)
        @test_nothrow Utils.clampcor(π)

        @test Utils.clampcor(BigFloat(3)) isa BigFloat
        @test Utils.clampcor(3.14) isa Float64
        @test Utils.clampcor(3.14f0) isa Float32
        @test Utils.clampcor(Float16(3)) isa Float16
        @test Utils.clampcor(BigInt(3)) isa BigInt
        @test Utils.clampcor(3) isa Int
        @test Utils.clampcor(3//1) isa Rational{Int}
        @test Utils.clampcor(π) isa Float64
    end
end

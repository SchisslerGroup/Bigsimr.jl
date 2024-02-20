using Test, Distributions, Bigsimr
using Bigsimr: _norm2margin, _randn, _rmvn, _idx_subsets2, _symmetric!, _set_diag1!,
    _clampcor


@testset "Internal Utilities" begin
    @testset "Normal to Marginal" begin
        D = Binomial(10, 0.5)

        # Must work for scalars, vectors, and matrices of type Float64
        @test_nowarn _norm2margin(D, randn())
        @test_nowarn _norm2margin(D, randn(2))
        @test_nowarn _norm2margin(D, randn(2, 2))
        @test_nowarn _norm2margin(D, randn(2, 2, 2))

        # Must work for other input types
        @test_nowarn _norm2margin(D, 1)
        @test_nowarn _norm2margin(D, 1//2)
        @test_nowarn _norm2margin(D, π)
        @test_nowarn _norm2margin(D, 0.57)
        @test_nowarn _norm2margin(D, 0.57f0)
        @test_nowarn _norm2margin(D, Float16(0.57))
        @test_nowarn _norm2margin(D, BigFloat(0.57))

        # Standard normal to standard normal should be invariant
        z = rand(Normal(0, 1), 100_000)
        @test z ≈ _norm2margin(Normal(0, 1), z)

        # Estimated parameters must be close to true parameters
        d1 = Binomial(20, 0.2)
        d2 = Poisson(3)
        d3 = Normal(12, π)

        x1 = _norm2margin(d1, z)
        x2 = _norm2margin(d2, z)
        x3 = _norm2margin(d3, z)

        f1 = fit_mle(Binomial, 20, x1)
        f2 = fit_mle(Poisson, x2)
        f3 = fit_mle(Normal, x3)

        @test all(isapprox.(params(d1), params(f1), rtol=0.01))
        @test all(isapprox.(params(d2), params(f2), rtol=0.01))
        @test all(isapprox.(params(d3), params(f3), rtol=0.01))
    end


    @testset "Random Normal" begin
        # General usage
        @test_nowarn _randn(Float64, 100, 10)
        @test_nowarn _randn(100, 10)

        # Must work for common floating-point types
        for T in (Float16, Float32, Float64)
            Z = _randn(T, 100_000, 10)
            @test eltype(Z) === T
        end
    end


    @testset "Random Multivariate Normal" begin
        r = Bigsimr._make_negdef_matrix()
        rho = cor_nearPD(r)

        # General usage
        @test_nowarn _rmvn(100, rho)
        @test_nowarn _rmvn(100, 0.5)
    end


    @testset "Other Helpers" begin
        X = reshape(collect(1:16), 4, 4)

        @test_nowarn _idx_subsets2(10)
        @test_nowarn _symmetric!(X)
        @test_nowarn _set_diag1!(X)

        @test_nowarn _clampcor(BigFloat(3))
        @test_nowarn _clampcor(3.14)
        @test_nowarn _clampcor(3.14f0)
        @test_nowarn _clampcor(Float16(3))
        @test_nowarn _clampcor(BigInt(3))
        @test_nowarn _clampcor(3)
        @test_nowarn _clampcor(3//1)
        @test_nowarn _clampcor(π)

        @test typeof(_clampcor(BigFloat(3))) === BigFloat
        @test typeof(_clampcor(3.14)) === Float64
        @test typeof(_clampcor(3.14f0)) === Float32
        @test typeof(_clampcor(Float16(3))) === Float16
        @test typeof(_clampcor(BigInt(3))) === BigInt
        @test typeof(_clampcor(3)) === Int
        @test typeof(_clampcor(3//1)) === Rational{Int}
        @test typeof(_clampcor(π)) === Float64
    end
end

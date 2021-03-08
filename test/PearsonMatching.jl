using Test
using Bigsimr
using Polynomials
using Distributions

@testset "Pearson Correlation Matching" begin
    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)
    
    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ pearson_match(-0.9, dA, dA) atol=0.05
        @test -0.611 ≈ pearson_match(-0.6, dA, dA) atol=0.05
        @test -0.306 ≈ pearson_match(-0.3, dA, dA) atol=0.05
        @test  0.304 ≈ pearson_match( 0.3, dA, dA) atol=0.05
        @test  0.606 ≈ pearson_match( 0.6, dA, dA) atol=0.05
        @test  0.904 ≈ pearson_match( 0.9, dA, dA) atol=0.05
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ pearson_match(-0.5, dB, dB) atol=0.05
        @test -0.501 ≈ pearson_match(-0.3, dB, dB) atol=0.05
        @test -0.322 ≈ pearson_match(-0.2, dB, dB) atol=0.05
        @test  0.418 ≈ pearson_match( 0.3, dB, dB) atol=0.05
        @test  0.769 ≈ pearson_match( 0.6, dB, dB) atol=0.05
        @test  0.944 ≈ pearson_match( 0.8, dB, dB) atol=0.05

        @test -0.939 ≈ pearson_match(-0.9, dC, dC) atol=0.05
        @test -0.624 ≈ pearson_match(-0.6, dC, dC) atol=0.05
        @test -0.311 ≈ pearson_match(-0.3, dC, dC) atol=0.05
        @test  0.310 ≈ pearson_match( 0.3, dC, dC) atol=0.05
        @test  0.618 ≈ pearson_match( 0.6, dC, dC) atol=0.05
        @test  0.925 ≈ pearson_match( 0.9, dC, dC) atol=0.05
    end

    @testset "Mixed" begin
        @test -0.890 ≈ pearson_match(-0.7, dB, dA) atol=0.05
        @test -0.632 ≈ pearson_match(-0.5, dB, dA) atol=0.05
        @test -0.377 ≈ pearson_match(-0.3, dB, dA) atol=0.05
        @test  0.366 ≈ pearson_match( 0.3, dB, dA) atol=0.05
        @test  0.603 ≈ pearson_match( 0.5, dB, dA) atol=0.05
        @test  0.945 ≈ pearson_match( 0.8, dB, dA) atol=0.05

        @test -0.928 ≈ pearson_match(-0.9, dC, dA) atol=0.05
        @test -0.618 ≈ pearson_match(-0.6, dC, dA) atol=0.05
        @test -0.309 ≈ pearson_match(-0.3, dC, dA) atol=0.05
        @test  0.308 ≈ pearson_match( 0.3, dC, dA) atol=0.05
        @test  0.613 ≈ pearson_match( 0.6, dC, dA) atol=0.05
        @test  0.916 ≈ pearson_match( 0.9, dC, dA) atol=0.05
    end

end

@testset "Pearson Correlation Utilities" begin

    @testset "Get Hermite Coefficients" begin
        dA = Binomial(20, 0.2)
        dB = NegativeBinomial(20, 0.002)
        dC = LogitNormal(3, 1)
        dD = Beta(5, 3)

        @test_nowarn Bigsimr.get_coefs(dA, 7)
        @test_nowarn Bigsimr.get_coefs(dB, 7)
        @test_nowarn Bigsimr.get_coefs(dC, 7)
        @test_nowarn Bigsimr.get_coefs(dD, 7)

        @test_nowarn Bigsimr.get_coefs(dA, 7.0)
        @test_nowarn Bigsimr.get_coefs(dB, 7.0)
        @test_nowarn Bigsimr.get_coefs(dC, 7.0)
        @test_nowarn Bigsimr.get_coefs(dD, 7.0)

        @test_throws InexactError Bigsimr.get_coefs(dA, 7.5)
        @test_throws InexactError Bigsimr.get_coefs(dB, 7.5)
        @test_throws InexactError Bigsimr.get_coefs(dC, 7.5)
        @test_throws InexactError Bigsimr.get_coefs(dD, 7.5)
    end

    @testset "Core Hermite Function" begin
        # Must work for any real input
        test_types = (Float64, Float32, Float16, BigFloat, Int128, Int64, Int32, Int16, BigInt, Rational)
        for T in test_types
            @test_nowarn Bigsimr._h(one(T), 5)
        end

        # For the following types, the input type should be the same as the output
        test_types = (Float64, Float32, Float16, BigFloat, Int64, BigInt, Rational)
        for T in test_types
            @test Bigsimr._h(one(T), 5) isa T
        end

        @test_nowarn Bigsimr._h(3.14, 5.0)
        @test_throws InexactError Bigsimr._h(3.14, 5.5)

        # Must work for arrays/matrices/vectors
        A = rand(3)
        B = rand(3, 3)
        C = rand(3, 3, 3)
        @test_nowarn Bigsimr._h(A, 3)
        @test_nowarn Bigsimr._h(B, 3)
        @test_nowarn Bigsimr._h(C, 3)
    end

    @testset "Hermite-Normal PDF" begin
        @test iszero(Bigsimr.Hp(Inf, 10))
        @test iszero(Bigsimr.Hp(-Inf, 10))
        @test 1.45182435 ≈ Bigsimr.Hp(1.0, 5)
    end

    @testset "Solve Polynomial on [-1, 1]" begin
        r1 = -1.0
        r2 = 1.0
        r3 = eps()
        r4 = 2 * rand() - 1

        P1 = coeffs(3 * fromroots([r1, 7, 7, 8]))
        P2 = coeffs(-5 * fromroots([r2, -1.14, -1.14, -1.14, -1.14, 1119]))
        P3 = coeffs(1.2 * fromroots([r3, nextfloat(1.0), prevfloat(-1.0)]))
        P4 = coeffs(fromroots([-5, 5, r4]))
        P5 = coeffs(fromroots([nextfloat(1.0), prevfloat(-1.0)]))
        P6 = coeffs(fromroots([-0.5, 0.5]))

        # One root at -1.0
        @test Bigsimr.solve_poly_pm_one(P1) ≈ r1 atol=0.001
        # One root at 1.0
        @test Bigsimr.solve_poly_pm_one(P2) ≈ r2 atol=0.001
        # Roots that are just outside [-1, 1]
        @test Bigsimr.solve_poly_pm_one(P3) ≈ r3 atol=0.001
        @test Bigsimr.solve_poly_pm_one(P4) ≈ r4 atol=0.001
        # Case of no roots
        @test isnan(Bigsimr.solve_poly_pm_one(P5))
        # Case of multiple roots
        @test length(Bigsimr.solve_poly_pm_one(P6)) == 2
        @test Bigsimr.nearest_root(-0.6, Bigsimr.solve_poly_pm_one(P6)) ≈ -0.5 atol=0.001
    end

end

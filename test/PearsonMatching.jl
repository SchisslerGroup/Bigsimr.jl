using Test
using MvSim
using Polynomials
using Distributions

@testset "Pearson Correlation Matching" begin

    @testset "Hermite-Normal PDF" begin
        @test iszero(MvSim.Hϕ(Inf, 10))
        @test iszero(MvSim.Hϕ(-Inf, 10))
        @test 1.45182435 ≈ MvSim.Hϕ(1.0, 5)
    end

    @testset "Solve Polynomial on [-1, 1]" begin
        r1 = -1.0
        r2 =  1.0
        r3 =  eps()
        r4 = 2 * rand() - 1

        P1 = coeffs(3 * fromroots([r1, 7, 7, 8]))
        P2 = coeffs(-5 * fromroots([r2, -1.14, -1.14, -1.14, -1.14, 1119]))
        P3 = coeffs(1.2 * fromroots([r3, nextfloat(1.0), prevfloat(-1.0)]))
        P4 = coeffs(fromroots([-5, 5, r4]))
        P5 = coeffs(fromroots([nextfloat(1.0), prevfloat(-1.0)]))
        P6 = coeffs(fromroots([-0.5, 0.5]))

        @test MvSim.solve_poly_pm_one(P1) ≈ r1 atol=0.001
        @test MvSim.solve_poly_pm_one(P2) ≈ r2 atol=0.001
        @test MvSim.solve_poly_pm_one(P3) ≈ r3 atol=0.001
        @test MvSim.solve_poly_pm_one(P4) ≈ r4 atol=0.001
        @test isnan(MvSim.solve_poly_pm_one(P5))
        @test_throws Exception MvSim.solve_poly_pm_one(P6)
    end

    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)
    
    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ pearson_match(-0.9, dA, dA, n=3) atol=0.01
        @test -0.611 ≈ pearson_match(-0.6, dA, dA, n=3) atol=0.01
        @test -0.306 ≈ pearson_match(-0.3, dA, dA, n=3) atol=0.01
        @test  0.304 ≈ pearson_match( 0.3, dA, dA, n=3) atol=0.01
        @test  0.606 ≈ pearson_match( 0.6, dA, dA, n=3) atol=0.01
        @test  0.904 ≈ pearson_match( 0.9, dA, dA, n=3) atol=0.01
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ pearson_match(-0.5, dB, dB, n=18) atol=0.01
        @test -0.501 ≈ pearson_match(-0.3, dB, dB, n= 3) atol=0.01
        @test -0.322 ≈ pearson_match(-0.2, dB, dB, n= 3) atol=0.01
        @test  0.418 ≈ pearson_match( 0.3, dB, dB, n= 3) atol=0.01
        @test  0.769 ≈ pearson_match( 0.6, dB, dB, n= 4) atol=0.01
        @test  0.944 ≈ pearson_match( 0.8, dB, dB, n=18) atol=0.01

        @test -0.939 ≈ pearson_match(-0.9, dC, dC) atol=0.01
        @test -0.624 ≈ pearson_match(-0.6, dC, dC) atol=0.01
        @test -0.311 ≈ pearson_match(-0.3, dC, dC) atol=0.01
        @test  0.310 ≈ pearson_match( 0.3, dC, dC) atol=0.01
        @test  0.618 ≈ pearson_match( 0.6, dC, dC) atol=0.01
        @test  0.925 ≈ pearson_match( 0.9, dC, dC) atol=0.01
    end

    @testset "Mixed" begin
        @test -0.890 ≈ pearson_match(-0.7, dB, dA) atol=0.01
        @test -0.632 ≈ pearson_match(-0.5, dB, dA) atol=0.01
        @test -0.377 ≈ pearson_match(-0.3, dB, dA) atol=0.01
        @test  0.366 ≈ pearson_match( 0.3, dB, dA) atol=0.01
        @test  0.603 ≈ pearson_match( 0.5, dB, dA) atol=0.01
        @test  0.945 ≈ pearson_match( 0.8, dB, dA) atol=0.01

        @test -0.928 ≈ pearson_match(-0.9, dC, dA) atol=0.01
        @test -0.618 ≈ pearson_match(-0.6, dC, dA) atol=0.01
        @test -0.309 ≈ pearson_match(-0.3, dC, dA) atol=0.01
        @test  0.308 ≈ pearson_match( 0.3, dC, dA) atol=0.01
        @test  0.613 ≈ pearson_match( 0.6, dC, dA) atol=0.01
        @test  0.916 ≈ pearson_match( 0.9, dC, dA) atol=0.01
    end

end

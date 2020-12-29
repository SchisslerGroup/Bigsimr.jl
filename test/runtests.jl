using MvSim
using Test
using Distributions
using Polynomials

import LinearAlgebra: eigvals, diag, isposdef

@testset "Utilities" begin

    @testset "Hermite Polynomials" begin
        He5(x) = x.^5 .- 10x.^3 .+ 15x     # Known Probabilists 5th degree
        H5(x) = 32x.^5 .- 160x.^3 .+ 120x  # Known Physicists 5th degree
        x = 200 * rand(100) .- 100
        @test all(MvSim.hermite.(x, 5) .≈ He5(x))
        @test all(MvSim.hermite.(x, 5, false) .≈ H5(x))
    end

    type_set = (Float64, Float32, Float16,
                Int64, Int32, Int16, Int8)
    @testset "A{$T1} → $T2" for T1 in type_set, T2 in type_set
        A = rand(T1, 4, 4)
        B = rand(T1, 4, 4)
        u = typemax(T2)
        l = typemin(T2)
        A = MvSim.setdiag(A, u)
        B = MvSim.setdiag(B, l)
        @test eltype(A) == promote_type(eltype(A), T2)
        @test eltype(B) == promote_type(eltype(B), T2)
        @test diag(A) == fill(eltype(A)(u), 4)
        @test diag(B) == fill(eltype(B)(l), 4)
    end

    @testset "Normal to Marginal" begin
        # Standard normal to standard normal should be invariant
        z = rand(Normal(0, 1), 100000)
        @test z ≈ MvSim.normal_to_margin(Normal(0, 1), z)

        d1 = Binomial(20, 0.2)
        d2 = Poisson(3)
        d3 = Normal(12, π)
        x1 = MvSim.normal_to_margin(d1, z)
        x2 = MvSim.normal_to_margin(d2, z)
        x3 = MvSim.normal_to_margin(d3, z)
        f1 = fit_mle(Binomial, 20, x1)
        f2 = fit_mle(Poisson, x2)
        f3 = fit_mle(Normal, x3)

        @test all(isapprox.(params(d1), params(f1), rtol=0.01))
        @test all(isapprox.(params(d2), params(f2), rtol=0.01))
        @test all(isapprox.(params(d3), params(f3), rtol=0.01))
    end

end


@testset "Correlation Utilities" begin

    @testset "Random postive definite correlation matrix" begin
        r = cor_randPD(Float64, 10)
        @test all(diag(r) .== 1.0)
        @test r == r'
        @test all(-1.0 .≤ r .≤ 1.0)
        λ = eigvals(r)
        @test all(λ .> 0)
        @test isposdef(r)
    end

    @testset "Random positive semi-definite correlation matrix" begin
        r = cor_randPSD(Float64, 10)
        @test all(diag(r) .== 1.0)
        @test r == r'
        @test all(-1.0 .≤ r .≤ 1.0)
        λ = eigvals(r)
        @test all(λ .≥ 0)
        @test isposdef(r)
    end

    @testset "Nearest positive definite correlation matrix" begin
        ρ = [
            1.00 0.82 0.56 0.44
            0.82 1.00 0.28 0.85
            0.56 0.28 1.00 0.22
            0.44 0.85 0.22 1.00
        ]

        ρ_hat = cor_nearPD(ρ)
        λ = eigvals(ρ_hat)
        @test all(λ .> 0)
        @test all(diag(ρ_hat) .== 1.0)
        @test ρ_hat ≈ ρ_hat' atol=1e-12
        @test all(-1.0 .≤ ρ_hat .≤ 1.0)
    end

    @testset "Nearest positive semi-definite correlation matrix" begin
        ρ = [
            1.00 0.82 0.56 0.44
            0.82 1.00 0.28 0.85
            0.56 0.28 1.00 0.22
            0.44 0.85 0.22 1.00
        ]

        ρ_hat = cor_nearPSD(ρ, n_iter=100)
        λ = eigvals(ρ_hat)
        # @test all(λ .≥ 0) # TODO: Fix or ignore this issue
        @test all(diag(ρ_hat) .== 1.0)
        @test ρ_hat ≈ ρ_hat' atol=1e-12
        @test all(-1.0 .≤ ρ_hat .≤ 1.0)
    end

    @testset "Correlation to correlation conversion" begin
        rs = cor_randPD(Float64, 4)
        rk = cor_randPD(Float64, 4)
        rp = cor_randPD(Float64, 4)
        rpp = cor_convert(rp, Pearson,  Pearson)
        rps = cor_convert(rp, Pearson,  Spearman)
        rpk = cor_convert(rp, Pearson,  Kendall)
        rsp = cor_convert(rs, Spearman, Pearson)
        rss = cor_convert(rs, Spearman, Spearman)
        rsk = cor_convert(rs, Spearman, Kendall)
        rkp = cor_convert(rk, Kendall,  Pearson)
        rks = cor_convert(rk, Kendall,  Spearman)
        rkk = cor_convert(rk, Kendall,  Kendall)

        @test rs == rss
        @test rk == rkk
        @test rp == rpp

        @test cor_convert(0.0, Spearman, Kendall)  ≈ 0.0
        @test cor_convert(0.0, Spearman, Pearson)  ≈ 0.0
        @test cor_convert(0.0, Kendall,  Spearman) ≈ 0.0
        @test cor_convert(0.0, Kendall,  Kendall)  ≈ 0.0
        @test cor_convert(0.0, Kendall,  Pearson)  ≈ 0.0
        @test cor_convert(0.0, Pearson,  Spearman) ≈ 0.0
        @test cor_convert(0.0, Pearson,  Kendall)  ≈ 0.0
        @test cor_convert(0.0, Pearson,  Pearson)  ≈ 0.0
        @test cor_convert(0.0, Spearman, Spearman) ≈ 0.0

        @test cor_convert(1.0, Spearman, Kendall)  ≈ 1.0
        @test cor_convert(1.0, Spearman, Pearson)  ≈ 1.0
        @test cor_convert(1.0, Kendall,  Spearman) ≈ 1.0
        @test cor_convert(1.0, Kendall,  Kendall)  ≈ 1.0
        @test cor_convert(1.0, Kendall,  Pearson)  ≈ 1.0
        @test cor_convert(1.0, Pearson,  Spearman) ≈ 1.0
        @test cor_convert(1.0, Pearson,  Kendall)  ≈ 1.0
        @test cor_convert(1.0, Pearson,  Pearson)  ≈ 1.0
        @test cor_convert(1.0, Spearman, Spearman) ≈ 1.0

        @test cor_convert(-1.0, Spearman, Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Spearman, Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Spearman) ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Spearman) ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Spearman, Spearman) ≈ -1.0
    end

end


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

        @test MvSim.solve_poly_pm_one(P1) ≈ r1 atol=0.0001
        @test MvSim.solve_poly_pm_one(P2) ≈ r2 atol=0.0001
        @test MvSim.solve_poly_pm_one(P3) ≈ r3 atol=0.0001
        @test MvSim.solve_poly_pm_one(P4) ≈ r4 atol=0.0001
        @test isnan(MvSim.solve_poly_pm_one(P5))
    end

    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)
    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ ρz(-0.9, dA, dA, n=3) atol=0.01
        @test -0.611 ≈ ρz(-0.6, dA, dA, n=3) atol=0.01
        @test -0.306 ≈ ρz(-0.3, dA, dA, n=3) atol=0.01
        @test  0.304 ≈ ρz( 0.3, dA, dA, n=3) atol=0.01
        @test  0.606 ≈ ρz( 0.6, dA, dA, n=3) atol=0.01
        @test  0.904 ≈ ρz( 0.9, dA, dA, n=3) atol=0.01
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ ρz(-0.5, dB, dB, n=18) atol=0.01
        @test -0.501 ≈ ρz(-0.3, dB, dB, n= 3) atol=0.01
        @test -0.322 ≈ ρz(-0.2, dB, dB, n= 3) atol=0.01
        @test  0.418 ≈ ρz( 0.3, dB, dB, n= 3) atol=0.01
        @test  0.769 ≈ ρz( 0.6, dB, dB, n= 4) atol=0.01
        @test  0.944 ≈ ρz( 0.8, dB, dB, n=18) atol=0.01

        @test -0.939 ≈ ρz(-0.9, dC, dC) atol=0.01
        @test -0.624 ≈ ρz(-0.6, dC, dC) atol=0.01
        @test -0.311 ≈ ρz(-0.3, dC, dC) atol=0.01
        @test  0.310 ≈ ρz( 0.3, dC, dC) atol=0.01
        @test  0.618 ≈ ρz( 0.6, dC, dC) atol=0.01
        @test  0.925 ≈ ρz( 0.9, dC, dC) atol=0.01
    end

    @testset "Mixed" begin
        @test -0.890 ≈ ρz(-0.7, dB, dA) atol=0.01
        @test -0.632 ≈ ρz(-0.5, dB, dA) atol=0.01
        @test -0.377 ≈ ρz(-0.3, dB, dA) atol=0.01
        @test  0.366 ≈ ρz( 0.3, dB, dA) atol=0.01
        @test  0.603 ≈ ρz( 0.5, dB, dA) atol=0.01
        @test  0.945 ≈ ρz( 0.8, dB, dA) atol=0.01

        @test -0.928 ≈ ρz(-0.9, dC, dA) atol=0.01
        @test -0.618 ≈ ρz(-0.6, dC, dA) atol=0.01
        @test -0.309 ≈ ρz(-0.3, dC, dA) atol=0.01
        @test  0.308 ≈ ρz( 0.3, dC, dA) atol=0.01
        @test  0.613 ≈ ρz( 0.6, dC, dA) atol=0.01
        @test  0.916 ≈ ρz( 0.9, dC, dA) atol=0.01
    end

end

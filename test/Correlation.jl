using Test
using bigsimr
using LinearAlgebra

@testset "Correlation Utilities" begin

    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]

    @testset "Random postive definite correlation matrix" begin
        r = cor_randPD(100)
        @test bigsimr.iscorrelation(r)
    end

    @testset "Random positive semi-definite correlation matrix" begin
        r = cor_randPSD(100)
        λ = eigvals(r)
        @test issymmetric(r)
        @test all(λ .≥ 0)
        @test all(diag(r) .== 1.0)
        @test all(-1.0 .≤ r .≤ 1.0)
    end

    @testset "Nearest positive definite correlation matrix" begin
        r = cor_nearPD(r_negdef)
        @test bigsimr.iscorrelation(r)
    end

    @testset "Nearest positive semi-definite correlation matrix" begin
        r = cor_nearPD(r_negdef, 0.0)
        λ = eigvals(r)
        @test issymmetric(r)
        @test all(λ .≥ 0)
        @test all(diag(r) .== 1.0)
        @test all(-1.0 .≤ r .≤ 1.0)
    end

    @testset "Correlation to correlation conversion" begin
        rs = cor_randPSD(4)
        rk = cor_randPSD(4)
        rp = cor_randPSD(4)
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
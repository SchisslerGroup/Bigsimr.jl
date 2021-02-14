using Test
using Bigsimr
using LinearAlgebra
using Distributions

@testset "Correlation Functions" begin
    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]

    @testset "Nearest positive definite correlation matrix" begin
        r = cor_nearPD(r_negdef)
        @test Bigsimr.iscorrelation(r)

        # Must respect input eltype
        for T in [Float64, Float32, Float64]
            @test eltype(cor_nearPD(Matrix{T}(r_negdef))) === T
        end
    end

    @testset "Nearest positive semi-definite correlation matrix" begin
        r = cor_nearPD(r_negdef, 0.0)
        λ = eigvals(r)
        @test issymmetric(r)
        @test all(λ .≥ 0)
        @test all(diag(r) .== 1.0)
        @test all(-1.0 .≤ r .≤ 1.0)
    end

    @testset "Fast near positive definite correlation matrix" begin
        r = cor_fastPD(r_negdef)
        @test Bigsimr.iscorrelation(r)

        # Must respect input eltype
        test_types = [Float64, Float32, Float16]
        for T in test_types
            @test eltype(cor_fastPD(Matrix{T}(r_negdef))) === T
        end
    end
end

@testset "Random Correlation Generation" begin
    
    @testset "Random postive definite correlation matrix" begin
        r = cor_randPD(100)
        @test Bigsimr.iscorrelation(r)

        # The element type must be respected
        test_types = [Float64, Float32, Float16]
        for T in test_types
            @test eltype(cor_randPD(T, 4)) === T
        end

        # Must work for numbers with integer representations
        test_types = [Float64, Float32, Float16, Rational, Int64, Int32, Int16]
        for T in test_types
            @test_nowarn cor_randPD(T(4))
            @test_nowarn cor_randPD(Float64, T(4))
            @test_nowarn cor_randPD(Float32, T(4))
            @test_nowarn cor_randPD(Float16, T(4))
            
            for S in test_types
                @test_nowarn cor_randPD(T(4), S(3))
                @test_nowarn cor_randPD(Float64, T(4), S(3))
                @test_nowarn cor_randPD(Float32, T(4), S(3))
                @test_nowarn cor_randPD(Float16, T(4), S(3))
            end
        end

        # `d` must not be less than 1
        @test_throws AssertionError cor_randPD(-1)
        @test_throws AssertionError cor_randPD(Float64, -1)
        @test_throws AssertionError cor_randPD(Float32, -1)
        @test_throws AssertionError cor_randPD(Float16, -1)

        # `k` must not be larger than `d`
        @test_throws AssertionError cor_randPD(4, 5)
        @test_throws AssertionError cor_randPD(Float64, 4, 5)
        @test_throws AssertionError cor_randPD(Float32, 4, 5)
        @test_throws AssertionError cor_randPD(Float16, 4, 5)

        # `k` must not be less than 1
        @test_throws AssertionError cor_randPD(4, 0)
        @test_throws AssertionError cor_randPD(Float64, 4, 0)
        @test_throws AssertionError cor_randPD(Float32, 4, 0)
        @test_throws AssertionError cor_randPD(Float16, 4, 0)
    end

    @testset "Random positive semi-definite correlation matrix" begin
        r = cor_randPSD(100)
        λ = eigvals(r)
        @test issymmetric(r)
        @test all(λ .≥ 0)
        @test all(diag(r) .== 1.0)
        @test all(-1.0 .≤ r .≤ 1.0)

        # The element type must be respected
        test_types = [Float64, Float32, Float16]
        for T in test_types
            @test eltype(cor_randPSD(T, 4)) === T
        end

        # Must work for numbers with integer representations
        test_types = [Float64, Float32, Float16, Rational, Int64, Int32, Int16]
        for T in test_types
            @test_nowarn cor_randPSD(T(4))
            @test_nowarn cor_randPSD(Float64, T(4))
            @test_nowarn cor_randPSD(Float32, T(4))
            @test_nowarn cor_randPSD(Float16, T(4))
            
            for S in test_types
                @test_nowarn cor_randPSD(T(4), S(3))
                @test_nowarn cor_randPSD(Float64, T(4), S(3))
                @test_nowarn cor_randPSD(Float32, T(4), S(3))
                @test_nowarn cor_randPSD(Float16, T(4), S(3))
            end
        end

        # `d` must not be less than 1
        @test_throws AssertionError cor_randPSD(-1)
        @test_throws AssertionError cor_randPSD(Float64, -1)
        @test_throws AssertionError cor_randPSD(Float32, -1)
        @test_throws AssertionError cor_randPSD(Float16, -1)

        # `k` must not be larger than `d`
        @test_throws AssertionError cor_randPSD(4, 5)
        @test_throws AssertionError cor_randPSD(Float64, 4, 5)
        @test_throws AssertionError cor_randPSD(Float32, 4, 5)
        @test_throws AssertionError cor_randPSD(Float16, 4, 5)

        # `k` must not be less than 1
        @test_throws AssertionError cor_randPSD(4, 0)
        @test_throws AssertionError cor_randPSD(Float64, 4, 0)
        @test_throws AssertionError cor_randPSD(Float32, 4, 0)
        @test_throws AssertionError cor_randPSD(Float16, 4, 0)
    end

end

@testset "Correlation Utilities" begin

    @testset "Correlation calculation" begin
        test_types = [Float64, Float32, Float16]
        for T in test_types
            A = rand(T, 200, 4)
            x, y = rand(T, 100), rand(T, 100)

            @test_nowarn cor(A, Pearson)
            @test_nowarn cor(A, Spearman)
            @test_nowarn cor(A, Kendall)

            @test_nowarn cor(x, y, Pearson)
            @test_nowarn cor(x, y, Spearman)
            @test_nowarn cor(x, y, Kendall)
        end

        test_types = [Float64, Float32, Float16]
        for T in test_types
            A = rand(T, 200, 4)

            @test_nowarn cor_fast(A, Pearson)
            @test_nowarn cor_fast(A, Spearman)
            @test_nowarn cor_fast(A, Kendall)
        end
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

        @test cor_convert(0.0, Spearman, Kendall)  == 0.0
        @test cor_convert(0.0, Spearman, Pearson)  == 0.0
        @test cor_convert(0.0, Kendall,  Spearman) == 0.0
        @test cor_convert(0.0, Kendall,  Kendall)  == 0.0
        @test cor_convert(0.0, Kendall,  Pearson)  == 0.0
        @test cor_convert(0.0, Pearson,  Spearman) == 0.0
        @test cor_convert(0.0, Pearson,  Kendall)  == 0.0
        @test cor_convert(0.0, Pearson,  Pearson)  == 0.0
        @test cor_convert(0.0, Spearman, Spearman) == 0.0

        @test cor_convert(1.0, Spearman, Kendall)  ≤ 1.0
        @test cor_convert(1.0, Spearman, Pearson)  ≤ 1.0
        @test cor_convert(1.0, Kendall,  Spearman) ≤ 1.0
        @test cor_convert(1.0, Kendall,  Kendall)  ≤ 1.0
        @test cor_convert(1.0, Kendall,  Pearson)  ≤ 1.0
        @test cor_convert(1.0, Pearson,  Spearman) ≤ 1.0
        @test cor_convert(1.0, Pearson,  Kendall)  ≤ 1.0
        @test cor_convert(1.0, Pearson,  Pearson)  ≤ 1.0
        @test cor_convert(1.0, Spearman, Spearman) ≤ 1.0

        @test cor_convert(1.0, Spearman, Kendall)  ≈ 1.0
        @test cor_convert(1.0, Spearman, Pearson)  ≈ 1.0
        @test cor_convert(1.0, Kendall,  Spearman) ≈ 1.0
        @test cor_convert(1.0, Kendall,  Kendall)  ≈ 1.0
        @test cor_convert(1.0, Kendall,  Pearson)  ≈ 1.0
        @test cor_convert(1.0, Pearson,  Spearman) ≈ 1.0
        @test cor_convert(1.0, Pearson,  Kendall)  ≈ 1.0
        @test cor_convert(1.0, Pearson,  Pearson)  ≈ 1.0
        @test cor_convert(1.0, Spearman, Spearman) ≈ 1.0

        @test cor_convert(-1.0, Spearman, Kendall)  ≥ -1.0
        @test cor_convert(-1.0, Spearman, Pearson)  ≥ -1.0
        @test cor_convert(-1.0, Kendall,  Spearman) ≥ -1.0
        @test cor_convert(-1.0, Kendall,  Kendall)  ≥ -1.0
        @test cor_convert(-1.0, Kendall,  Pearson)  ≥ -1.0
        @test cor_convert(-1.0, Pearson,  Spearman) ≥ -1.0
        @test cor_convert(-1.0, Pearson,  Kendall)  ≥ -1.0
        @test cor_convert(-1.0, Pearson,  Pearson)  ≥ -1.0
        @test cor_convert(-1.0, Spearman, Spearman) ≥ -1.0

        @test cor_convert(-1.0, Spearman, Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Spearman, Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Spearman) ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Kendall,  Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Spearman) ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Kendall)  ≈ -1.0
        @test cor_convert(-1.0, Pearson,  Pearson)  ≈ -1.0
        @test cor_convert(-1.0, Spearman, Spearman) ≈ -1.0

        # Must work for each type. Must respect the input eltype
        test_types = [Float64, Float32, Float16, Rational]
        for T in test_types
            @test_nowarn cor_convert(T(0.5), Pearson,  Pearson)
            @test_nowarn cor_convert(T(0.5), Pearson,  Spearman)
            @test_nowarn cor_convert(T(0.5), Pearson,  Kendall)
            @test_nowarn cor_convert(T(0.5), Spearman, Pearson)
            @test_nowarn cor_convert(T(0.5), Spearman, Spearman)
            @test_nowarn cor_convert(T(0.5), Spearman, Kendall)
            @test_nowarn cor_convert(T(0.5), Kendall,  Pearson)
            @test_nowarn cor_convert(T(0.5), Kendall,  Spearman)
            @test_nowarn cor_convert(T(0.5), Kendall,  Kendall)
        end

        test_types = [Float64, Float32, Float16]
        for T in test_types
            @test eltype(cor_convert(T(0.5), Pearson,  Pearson))  === T
            @test eltype(cor_convert(T(0.5), Pearson,  Spearman)) === T
            @test eltype(cor_convert(T(0.5), Pearson,  Kendall))  === T
            @test eltype(cor_convert(T(0.5), Spearman, Pearson))  === T
            @test eltype(cor_convert(T(0.5), Spearman, Spearman)) === T
            @test eltype(cor_convert(T(0.5), Spearman, Kendall))  === T
            @test eltype(cor_convert(T(0.5), Kendall,  Pearson))  === T
            @test eltype(cor_convert(T(0.5), Kendall,  Spearman)) === T
            @test eltype(cor_convert(T(0.5), Kendall,  Kendall))  === T
        end
    end

    @testset "Constrain to Correlation" begin
        test_types = [Float64, Float32, Float16]
        for T in test_types
            A = rand(T, 10, 10)
            @test_nowarn cor_constrain(A)
            @test eltype(cor_constrain(A)) === T
        end
    end

    @testset "Covariance to Correlation" begin
        test_types = [Float64, Float32, Float16]
        for T in test_types
            R = cor_randPSD(T, 4, 4)
            @test_nowarn cov2cor(R)
            @test eltype(cov2cor(R)) === T
        end
    end

    @testset "Correlation Bounds" begin
        A, B = NegativeBinomial(20, 0.2), LogNormal(3, 1)

        @test_nowarn cor_bounds(A, A, Pearson)
        @test_nowarn cor_bounds(A, A, Spearman)
        @test_nowarn cor_bounds(A, A, Kendall)

        @test_nowarn cor_bounds(A, B, Pearson)
        @test_nowarn cor_bounds(A, B, Spearman)
        @test_nowarn cor_bounds(A, B, Kendall)

        @test_nowarn cor_bounds(B, A, Pearson)
        @test_nowarn cor_bounds(B, A, Spearman)
        @test_nowarn cor_bounds(B, A, Kendall)

        test_types = [Float64, Float32, Float16, Int64, Int32, Int16]
        for T in test_types
            @test_nowarn cor_bounds(A, B, n_samples=T(10_000))
        end

        test_types = [Float64, Float32]
        for T in test_types
            @test_throws InexactError cor_bounds(A, B, n_samples=T(10_000.5))
        end 
    end

end
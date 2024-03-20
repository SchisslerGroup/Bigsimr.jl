using Test, Bigsimr
using Distributions


include("test_macros.jl")


@testset "Correlations" begin
    x = rand(100)
    y = rand(100)
    M = rand(100, 10)

    for cortype in (Pearson, Spearman, Kendall)
        @test_nothrow cor(x, y, cortype)
        @test_nothrow cor(M,    cortype)
        @test_nothrow cor_fast(M, cortype)
    end

    x = 0.5
    xs = rand(10)
    r = cor_nearPD(Bigsimr._make_negdef_matrix())

    for cortype1 in (Pearson, Spearman, Kendall)
        for cortype2 in (Pearson, Spearman, Kendall)
            @test_nothrow cor_convert(x, cortype1, cortype2)
            @test_nothrow cor_convert(xs, cortype1, cortype2)
            @test_nothrow cor_convert(r, cortype1, cortype2)
            if cortype1 === cortype2
                @test cor_convert(x, cortype1, cortype2) == x
            end
        end
    end

    for T in (Float16, Float32, Float64)
        X = rand(T, 10, 10)
        @test_nothrow cor_constrain!(X)
        @test eltype(X) === T
    end

    for T in (Float16, Float32, Float64)
        X = T.(cor_nearPD(Bigsimr._make_negdef_matrix()))
        @test_nothrow cov2cor(X)
        Y = cov2cor(X)
        @test eltype(Y) === T
        @test_nothrow cov2cor!(X)
        @test eltype(X) === T
    end

    for T in (Float16, Float32, Float64)
        @test_nothrow cor_randPSD(T, 10, 4)
        @test_nothrow cor_randPSD(10, 4)
        @test_throws Exception cor_randPSD(T, 3, 4)
        @test_throws Exception cor_randPSD(3, 4)
        @test_throws Exception cor_randPSD(T, 3.4, 4.2)
        @test_throws Exception cor_randPSD(T, 3, 4.2)
        @test_throws Exception cor_randPSD(T, 3.4, 4)
        @test eltype(cor_randPSD(T, 10)) === T

        @test_nothrow cor_randPD(T, 10, 4)
        @test_nothrow cor_randPD(10, 4)
        @test_throws Exception cor_randPD(T, 3, 4)
        @test_throws Exception cor_randPD(3, 4)
        @test_throws Exception cor_randPD(T, 3.4, 4.2)
        @test_throws Exception cor_randPD(T, 3, 4.2)
        @test_throws Exception cor_randPD(T, 3.4, 4)
        @test eltype(cor_randPD(T, 10)) === T
    end

    d1 = Normal()
    d2 = NegativeBinomial(20, 0.3)
    d3 = Gamma()
    margins = [d1, d2, d3]

    for cortype in (Pearson, Spearman, Kendall)
        @test_nothrow cor_bounds(d1, d2, cortype)
        @test_nothrow cor_bounds(margins, cortype)
    end

    for T in (Float16, Float32, Float64)
        r = T.(Bigsimr._make_negdef_matrix())

        @test_nothrow cor_nearPD(r)
        @test_nothrow cor_nearPSD(r)
        @test_nothrow cor_fastPD(r)

        @test eltype(cor_nearPD(r)) === T
        @test eltype(cor_nearPSD(r)) === T
        @test eltype(cor_fastPD(r)) === T
    end
end

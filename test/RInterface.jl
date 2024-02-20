using Test, Bigsimr

#=
One of the strengths of this package is that it is intended to be compatible with R. Since
R doesn't really have the concept of integer numbers or scalars, we need to ensure that
the public methods work with floating point numbers that can be represented as integers.
E.g. rmvn(100.0, Σ) should work because 100.0 can be represented as an integer.
=#


@testset "R Interface" begin
    @test_nowarn cor_randPSD(3.0)
    @test_nowarn cor_randPSD(6.0, 3.0)
    @test_nowarn cor_randPD(3.0)
    @test_nowarn cor_randPD(6.0, 3.0)

    d1 = Normal()
    d2 = NegativeBinomial(20, 0.3)
    d3 = Gamma()
    margins = [d1, d2, d3]

    @test_nowarn cor_bounds(d1, d2, Pearson, 100000.0)
    @test_nowarn cor_bounds(margins, Pearson, 100000.0)
    @test_nowarn cor_bounds(d1, d2, 100000.0)
    @test_nowarn cor_bounds(margins, 100000.0)

    d = length(margins)
    Σ = cor_randPD(d)
    μ = rand(d)
    @test_nowarn rmvn(100.0, μ, Σ)
    @test_nowarn rmvn(100.0,    Σ)

    @test_nowarn rvec(100.0, Σ, margins)
end

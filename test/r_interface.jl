using Test, Bigsimr
using Distributions


include("test_macros.jl")


#=
One of the strengths of this package is that it is intended to be compatible with R. Since
R doesn't really have the concept of integer numbers or scalars, we need to ensure that
the public methods work with floating point numbers that can be represented as integers.
E.g. rmvn(100.0, Σ) should work because 100.0 can be represented as an integer.
=#


@testset "R Interface" begin
    # methods that are expected to be used from R
    @test_isdefined cor
    @test_isdefined cor_fast
    @test_isdefined cor_bounds
    @test_isdefined cor_convert
    @test_isdefined cor_constrain
    @test_isdefined cov2cor
    @test_isdefined is_correlation

    @test_isdefined cor_randPSD
    @test_isdefined cor_randPD

    @test_isdefined cor_nearPSD
    @test_isdefined cor_nearPD
    @test_isdefined cor_fastPD
    @test_isdefined nearest_cor

    @test_isdefined rmvn
    @test_isdefined rvec

    @test_isdefined pearson_match
    @test_isdefined pearson_bounds

    d1 = Normal()
    d2 = NegativeBinomial(20, 0.3)
    d3 = Gamma()
    margins = [d1, d2, d3]
    d = length(margins)
    Σ = cor_randPD(d)
    μ = rand(d)

    # methods should be implemented for int-like numbers
    @test_isimplemented cor_randPSD(3.0)
    @test_isimplemented cor_randPSD(6.0, 3.0)
    @test_isimplemented cor_randPD(3.0)
    @test_isimplemented cor_randPD(6.0, 3.0)
    @test_isimplemented cor_bounds(d1, d2, Pearson, 100000.0)
    @test_isimplemented cor_bounds(margins, Pearson, 100000.0)
    @test_isimplemented cor_bounds(d1, d2, 100000.0)
    @test_isimplemented cor_bounds(margins, 100000.0)
    @test_isimplemented rmvn(100.0, μ, Σ)
    @test_isimplemented rmvn(100.0,    Σ)
    @test_isimplemented rvec(100.0, Σ, margins)

    # methods should work without error for int-like numbers
    @test_nothrow cor_randPSD(3.0)
    @test_nothrow cor_randPSD(6.0, 3.0)
    @test_nothrow cor_randPD(3.0)
    @test_nothrow cor_randPD(6.0, 3.0)
    @test_nothrow cor_bounds(d1, d2, Pearson, 100000.0)
    @test_nothrow cor_bounds(margins, Pearson, 100000.0)
    @test_nothrow cor_bounds(d1, d2, 100000.0)
    @test_nothrow cor_bounds(margins, 100000.0)
    @test_nothrow rmvn(100.0, μ, Σ)
    @test_nothrow rmvn(100.0,    Σ)
    @test_nothrow rvec(100.0, Σ, margins)
end

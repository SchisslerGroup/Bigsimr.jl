using Test, Bigsimr
using Distributions


include("test_macros.jl")


@testset "Random Vector Generation" begin
    d = 3
    Σ = cor_randPD(d)
    μ = rand(d)
    @test_nothrow rmvn(100, μ, Σ)
    @test_nothrow rmvn(100,    Σ)

    # mean vector and covariance matrix dimensions must agree
    @test_throws Exception rmvn(100, rand(4), cor_randPD(3))
    # covariance matrix must be valid
    @test_throws Exception rmvn(100, rand(4, 4))
    @test_throws Exception rmvn(100, rand(4, 5))

    margins = [Normal(3, 1), LogNormal(3, 1), Exponential(3)]
    @test_nothrow rvec(100, Σ, margins)
end

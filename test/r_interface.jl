using Test, Bigsimr
using Distributions
using Distributions: UnivariateDistribution as UD

include("test_macros.jl")

#=
One of the strengths of this package is that it is intended to be compatible with R. Since
R doesn't really have the concept of integer numbers or scalars, we need to ensure that
the public methods work with floating point numbers that can be represented as integers.
E.g. rmvn(100.0, Σ) should work because 100.0 can be represented as an integer.
=#

# Do not remove any symbols from this list. They are considered locked
const r_api_symbols = [
    :cor,
    :cor_fast,
    :cor_bounds,
    :cor_convert,
    :cor_constrain,
    :cov2cor,
    :is_correlation,
    :cor_randPSD,
    :cor_randPD,
    :cor_nearPSD,
    :cor_nearPD,
    :cor_fastPD,
    :nearest_cor,
    :rmvn,
    :rvec,
    :pearson_match,
    :pearson_bounds,
    :Pearson,
    :Spearman,
    :Kendall
];

@testset verbose = true "R Interface" begin
    @testset failfast = true "API symbols defined" begin
        for f in r_api_symbols
            @eval @test_isdefined $f
        end
    end

    @testset failfast = true "Support valid R input" begin
        @test_hasmethod cor (CorType, AbstractVecOrMat{<:Real})
        @test_hasmethod cor (CorType, AbstractVecOrMat{<:Real}, AbstractVecOrMat{<:Real})
        @test_hasmethod cor_fast (CorType, AbstractMatrix{<:Real})

        @test_hasmethod cor_bounds (UD, UD, CorType, Real)
        @test_hasmethod cor_bounds (UD, UD, Real)
        @test_hasmethod cor_bounds (AbstractVector{UD}, CorType, Real)
        @test_hasmethod cor_bounds (AbstractVector{UD}, Real)

        @test_hasmethod cor_convert (Real, CorType, CorType)
        @test_hasmethod cor_convert (AbstractVector{<:Real}, CorType, CorType)
        @test_hasmethod cor_convert (AbstractMatrix{<:Real}, CorType, CorType)
        @test_hasmethod cor_constrain (AbstractMatrix{<:Real},)
        @test_hasmethod cov2cor (AbstractMatrix{<:Real},)
        @test_hasmethod is_correlation (AbstractMatrix{<:Real},)

        @test_hasmethod cor_randPSD (Real,)
        @test_hasmethod cor_randPSD (Real, Real)

        @test_hasmethod cor_randPD (Real,)
        @test_hasmethod cor_randPD (Real, Real)

        @test_hasmethod cor_nearPSD (AbstractMatrix{<:Real},)
        @test_hasmethod cor_nearPD (AbstractMatrix{<:Real},)
        @test_hasmethod cor_fastPD (AbstractMatrix{<:Real},)
        @test_hasmethod nearest_cor (AbstractMatrix{<:Real},)

        @test_hasmethod rmvn (Real, AbstractVector{<:Real}, AbstractMatrix{<:Real})
        @test_hasmethod rmvn (Real, AbstractMatrix{<:Real})
        @test_hasmethod rvec (Real, AbstractMatrix{<:Real}, AbstractVector{UD})

        @test_hasmethod pearson_match (Real, UD, UD)
        @test_hasmethod pearson_match (AbstractMatrix{<:Real}, AbstractVector{UD})

        @test_hasmethod pearson_bounds (UD, UD)
        @test_hasmethod pearson_bounds (AbstractVector{UD},)
    end

    # methods should work without error for int-like numbers
    @testset "Methods work with valid R input" begin
        d1 = Normal()
        d2 = NegativeBinomial(20, 0.3)
        d3 = Gamma()
        margins = [d1, d2, d3]
        d = length(margins)
        Σ = cor_randPD(d)
        μ = rand(d)

        @test_nothrow cor_randPSD(3.0)
        @test_nothrow cor_randPSD(6.0, 3.0)
        @test_nothrow cor_randPD(3.0)
        @test_nothrow cor_randPD(6.0, 3.0)
        @test_nothrow cor_bounds(d1, d2, Pearson, 100000.0)
        @test_nothrow cor_bounds(margins, Pearson, 100000.0)
        @test_nothrow cor_bounds(d1, d2, 100000.0)
        @test_nothrow cor_bounds(margins, 100000.0)
        @test_nothrow rmvn(100.0, μ, Σ)
        @test_nothrow rmvn(100.0, Σ)
        @test_nothrow rvec(100.0, Σ, margins)
    end
end

using Test
using Bigsimr
using Distributions

@testset "Generalized S-Distribution" begin
    D = Gamma(2, 2)
    G = Bigsimr.GSDist(D)

    @testset "Parameter retrieval" begin
        @test_nowarn params(G)
    end

    @testset "Computation of statistics" begin
        @test_nowarn mean(G)
        @test_nowarn var(G)
        @test_nowarn std(G)
        @test_nowarn median(G)
    end

    @testset "Probability evaluation" begin
        x = median(G)
        y = median(G) + 1

        @test_nowarn quantile(G, 0.3)
        @test_nowarn cquantile(G, 0.3)
    end

    @testset "Sampling" begin
        @test_nowarn sampler(G)
        @test_nowarn rand(G, 10)
    end

    @testset "Pearson Matching w/ GSDist" begin
        dA = NegativeBinomial(20, 0.1)
        dB = Gamma(100, 4)

        # Converting to GSDist must warn the user
        msg = "$dA was converted to a GSDist for computational efficiency"
        @test_logs (:warn, msg) (:warn, msg) pearson_match(-0.9, dA, dA)
        @test_logs (:warn, msg)              pearson_match(-0.9, dA, dB)
        @test_logs (:warn, msg)              pearson_match(-0.9, dB, dA)

        # Not converting does not warn the user
        @test_nowarn pearson_match(-0.9, dB, dB)
        @test_nowarn pearson_match(-0.9, dA, dA, convert=false)
        @test_nowarn pearson_match(-0.9, dA, dB, convert=false)
        @test_nowarn pearson_match(-0.9, dB, dA, convert=false)
        @test_nowarn pearson_match(-0.9, dB, dB, convert=false)

        # Converting must retain accuracy
        @test pearson_match(-0.9, dA, dA) ≈ pearson_match(-0.9, dA, dA, convert=false) atol=0.05
        # @test pearson_match(-0.9, dA, dB) ≈ pearson_match(-0.9, dA, dB, convert=false) atol=0.05
        # @test pearson_match(-0.9, dB, dA) ≈ pearson_match(-0.9, dB, dA, convert=false) atol=0.05
        @test pearson_match(-0.9, dB, dB) ≈ pearson_match(-0.9, dB, dB, convert=false) atol=0.05
    end

end

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

    # @testset "Pearson Matching w/ GSDist" begin
    #     d1 = NegativeBinomial(20, 0.1)
    #     d2 = Gamma(100, 4)

    #     @test pearson_match(-0.9, d1, d1; convert=true) ≈ pearson_match(-0.9, d1, d1; convert=false) atol=0.05
    #     @test pearson_match(-0.9, d1, d2; convert=true) ≈ pearson_match(-0.9, d1, d2; convert=false) atol=0.05
    #     @test pearson_match(-0.9, d2, d1; convert=true) ≈ pearson_match(-0.9, d2, d1; convert=false) atol=0.05
    #     @test pearson_match(-0.9, d2, d2; convert=true) ≈ pearson_match(-0.9, d2, d2; convert=false) atol=0.05
    # end

end

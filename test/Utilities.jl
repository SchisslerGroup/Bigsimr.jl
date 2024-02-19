using Test
using Bigsimr
using Distributions
using IrrationalConstants: quartπ

@testset "Utilities" begin

    @testset "Special Functions" begin
        for T in (Float64, Float32, Float16, BigFloat)
            # Must return same floating point type as input
            @test typeof(Bigsimr._normpdf(T(1.0))) === T
            @test typeof(Bigsimr._normcdf(T(1.0))) === T
            @test typeof(Bigsimr._norminvcdf(T(0.5))) === T
        end

        # Integer inputs must return Float64
        @test typeof(Bigsimr._normpdf(1)) === Float64
        @test typeof(Bigsimr._normcdf(1)) === Float64
        @test typeof(Bigsimr._norminvcdf(0)) === Float64
        # Ratoinal inputs must return Float64
        @test typeof(Bigsimr._normpdf(1//2)) === Float64
        @test typeof(Bigsimr._normcdf(3//4)) === Float64
        @test typeof(Bigsimr._norminvcdf(3//5)) === Float64
        # Irrational inputs must return Float64
        @test typeof(Bigsimr._normpdf(π)) === Float64
        @test typeof(Bigsimr._normcdf(π)) === Float64
        @test typeof(Bigsimr._norminvcdf(quartπ)) === Float64
    end

end

using Test
using Bigsimr
using Distributions

@testset "Utilities" begin

    @testset "Special Functions" begin
        test_types = (Float64, Float32, Float16)
        for T in test_types
            # Must work for each type
            @test_nowarn Bigsimr._normpdf(T(1.0))
            @test_nowarn Bigsimr._normcdf(T(1.0))
            @test_nowarn Bigsimr._norminvcdf(T(0.5))

            # Must respect floating point types
            @test typeof(Bigsimr._normpdf(T(1.0))) === T
            @test typeof(Bigsimr._normcdf(T(1.0))) === T
            @test typeof(Bigsimr._norminvcdf(T(0.5))) === T
        end
    end

end

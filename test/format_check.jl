using Test
using Bigsimr
using JuliaFormatter

if VERSION >= v"1.6"
    @test JuliaFormatter.format(Bigsimr; verbose=false, overwrite=false)
end

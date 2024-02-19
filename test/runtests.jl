using Test, Bigsimr
using Aqua


Aqua.test_all(Bigsimr; ambiguities=false)
Aqua.test_ambiguities(Bigsimr)


const tests = [
    "Correlation",
    "PearsonMatching",
    "GeneralizedSDistribution",
    "RandomVector",
    "Utilities"
]

printstyled("Running tests:\n", color=:blue)

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end

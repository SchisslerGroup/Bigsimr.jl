using Test
using Aqua


include("Internals.jl")
include("Correlations.jl")
include("RandomVectors.jl")
include("RInterface.jl")


Aqua.test_all(Bigsimr; ambiguities=false)
Aqua.test_ambiguities(Bigsimr)

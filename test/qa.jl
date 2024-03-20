using Bigsimr, Aqua

@testset "Quality Assurance" begin
    Aqua.test_all(Bigsimr; ambiguities=false)
    Aqua.test_ambiguities(Bigsimr)
end

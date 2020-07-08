using Documenter, MvSim

makedocs(
    sitename="MvSim.jl",
    modules = [MvSim],
    doctest=false
)

deploydocs(
    repo = "github.com/adknudson/MvSim.jl.git",
    versions = ["stable" => "master", "dev" => "develop"]
)

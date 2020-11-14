push!(LOAD_PATH,"../src/")
using Documenter, MvSim
using Distributions

makedocs(
    sitename = "MvSim.jl",
    modules  = [MvSim],
    doctest  = false
)

deploydocs(
    repo = "github.com/adknudson/MvSim.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

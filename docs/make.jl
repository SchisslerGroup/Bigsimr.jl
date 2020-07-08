using Documenter, MvSim

makedocs(
    sitename="MvSim.jl",
    modules = [MvSim],
    doctest=false
)

deploydocs(
    repo = "github.com/adknudson/MvSim.jl.git",
    devbranch = "develop"
)

using Documenter, MvSim

makedocs(
    sitename="MvSim Docs",
    doctest=false,
    clean=true
)

deploydocs(
    repo = "github.com/adknudson/MvSim.jl.git",
    devurl = "develop"
)

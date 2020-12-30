push!(LOAD_PATH,"../src/")
using Documenter, MvSim
using Distributions

makedocs(
    sitename = "MvSim.jl",
    modules  = [MvSim],
    doctest  = false,
    pages    = [
        "MvSim.jl" => "index.md",
        "Guides" => [
            "Getting Started" => "tutorial_getting_started.md",
            "Pearson Matching" => "tutorial_pearson_matching.md"
        ],
        "Main Functions" => "main_functions.md",
        "Utilities" => "utilities.md",
        "Details" => "details.md",
        "Index" => "function_index.md"
    ]
)

deploydocs(
    repo = "github.com/adknudson/MvSim.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

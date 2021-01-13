push!(LOAD_PATH,"../src/")
using Documenter, MvSim
using Distributions

DocMeta.setdocmeta!(MvSim, :DocTestSetup, :(using MvSim); recursive=true)

makedocs(
    sitename = "MvSim.jl",
    modules  = [MvSim],
    pages    = [
        "MvSim.jl" => "index.md",
        "Guides" => [
            "Getting Started" => "getting_started.md",
            "Pearson Matching" => "pearson_matching.md",
            "Nearest Correlation Matrix" => "nearest_correlation_matrix.md"
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

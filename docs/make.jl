using Documenter, Bigsimr
using Distributions

DocMeta.setdocmeta!(Bigsimr, :DocTestSetup, :(using Bigsimr); recursive=true)

makedocs(
    sitename = "Bigsimr.jl",
    modules  = [Bigsimr],
    doctest  = false,
    pages    = [
        "Bigsimr.jl" => "index.md",
        "Guides" => [
            "Getting Started" => "getting_started.md",
            "Pearson Matching" => "pearson_matching.md",
            "Nearest Correlation Matrix" => "nearest_correlation_matrix.md"
        ],
        "API Reference" => "main_functions.md",
        "Details" => "details.md",
        "Index" => "function_index.md"
    ]
)

deploydocs(
    repo = "github.com/SchisslerGroup/Bigsimr.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

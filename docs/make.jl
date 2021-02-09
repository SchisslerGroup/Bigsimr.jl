using Documenter, bigsimr
using Distributions

DocMeta.setdocmeta!(bigsimr, :DocTestSetup, :(using bigsimr); recursive=true)

makedocs(
    sitename = "bigsimr.jl",
    modules  = [bigsimr],
    pages    = [
        "bigsimr.jl" => "index.md",
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
    repo = "github.com/adknudson/bigsimr.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

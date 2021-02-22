using Documenter, Bigsimr
using Distributions

DocMeta.setdocmeta!(Bigsimr, :DocTestSetup, :(using Bigsimr); recursive=true)

makedocs(
    sitename = "Bigsimr.jl",
    modules  = [Bigsimr],
    pages    = [
        "Bigsimr.jl" => "index.md",
        "Guides" => [
            "Getting Started" => "getting_started.md",
            "Pearson Matching" => "pearson_matching.md",
            "Nearest Correlation Matrix" => "nearest_correlation_matrix.md"
        ],
        "Main Functions" => "main_functions.md",
        "Details" => "details.md",
        "Index" => "function_index.md"
    ]
)

deploydocs(
    repo = "github.com/adknudson/Bigsimr.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

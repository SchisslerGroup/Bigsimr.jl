using Documenter, Bigsimr
using Distributions

DocMeta.setdocmeta!(Bigsimr, :DocTestSetup, :(using Bigsimr); recursive=true)

# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

makedocs(
    sitename = "Bigsimr.jl",
    modules = [Bigsimr, PearsonCorrelationMatch],
    format = Documenter.HTML(),
    doctest = false,
    checkdocs = :exports,
    pages = [
        "Bigsimr.jl" => "index.md",
        "Guides" => [
            "Getting Started" => "getting_started.md",
            "Pearson Matching" => "pearson_matching.md",
            "Nearest Correlation Matrix" => "nearest_correlation_matrix.md"
        ],
        "API Reference" => "main_functions.md",
        "Details" => "details.md",
        "Index" => "function_index.md"
    ],
)

deploydocs(
    repo = "github.com/SchisslerGroup/Bigsimr.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)

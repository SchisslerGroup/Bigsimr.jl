using SafeTestsets

@safetestset "Quality Assurance" include("qa.jl")
@safetestset "Code Formatting" include("format_check.jl")
@safetestset "Internals" include("internals.jl")
@safetestset "Correlation Utils" include("correlations.jl")
@safetestset "Random Gen Utils" include("random_vectors.jl")
@safetestset "R Interface" include("r_interface.jl")

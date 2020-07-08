var documenterSearchIndex = {"docs":
[{"location":"utilities/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utilities/","page":"Utilities","title":"Utilities","text":"rcor(d::Integer, k::Integer=1)\ncov2cor(rho::AbstractArray)","category":"page"},{"location":"utilities/#MvSim.rcor","page":"Utilities","title":"MvSim.rcor","text":"rcor(d::Integer, k::Integer=1)\n\nGenerate a random correlation matrix of size dd. the parameter k is used to set the factor loadings for W. The code has been adapted from user amoeba from StackExchange\n\n\n\n\n\n","category":"function"},{"location":"utilities/#MvSim.cov2cor-Tuple{AbstractArray}","page":"Utilities","title":"MvSim.cov2cor","text":"cov2cor(rho::AbstractArray)\n\nConvert a covariance matrix to a correlation matrix. Ensure that the resulting matrix is symmetric and has diagonals equal to 1.0.\n\n\n\n\n\n","category":"method"},{"location":"#MvSim.jl-Documentation","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"","category":"section"},{"location":"","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"Pages = [\"main_functions.md\", \"utilities.md\"]\n","category":"page"},{"location":"#Index","page":"MvSim.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"","category":"page"},{"location":"main_functions/#Main-Functions","page":"Main Functions","title":"Main Functions","text":"","category":"section"},{"location":"main_functions/","page":"Main Functions","title":"Main Functions","text":"nearestPSDcor(R)","category":"page"},{"location":"main_functions/#MvSim.nearestPSDcor-Tuple{Any}","page":"Main Functions","title":"MvSim.nearestPSDcor","text":"nearestPSDcor(R)\n\nCompute the nearest positive semidefinite correlation matrix given a symmetric correlation matrix R. This algorithm is based off of work by Qi and Sun 2006. Matlab, C, R, and Python code can be found on Sun's page. The algorithm has also been implemented in Fortran in the NAG library.\n\nArguments\n\nτ::Real: a [small] nonnegative number used to enforce a minimum eigenvalue.\nδ::Real: the error tolerance for the stopping condition.\n\nExamples\n\nimport LinearAlgebra: eigvals\n# Define a negative definite correlation matrix\nρ = [1.00 0.82 0.56 0.44\n     0.82 1.00 0.28 0.85\n     0.56 0.28 1.00 0.22\n     0.44 0.85 0.22 1.00]\neigvals(ρ)\n\nr = nearestPSDcor(ρ)\neigvals(r)\n\n\n\n\n\n","category":"method"}]
}

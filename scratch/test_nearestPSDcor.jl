using MvSim
using DataFrames

import CSV
import LinearAlgebra


ρ1k = DataFrame!(CSV.File(
    "/home/alex/projects/SchisslerGroup@github/bigsimr/scratch/rho_ND_1K.csv",
    header=false)) |>
    Matrix{Float64}

LinearAlgebra.isposdef(ρ1k)

r1k = cor_nearPSD(ρ1k)
Juno.@enter cor_nearPSD(ρ1k)

LinearAlgebra.isposdef(r1k)
sum(r1k .- ρ1k)
LinearAlgebra.norm2(r1k .- ρ1k)
LinearAlgebra.issymmetric(r1k)

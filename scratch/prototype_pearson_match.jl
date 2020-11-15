using MvSim
using Distributions

import Base.Threads: @threads
import IterTools: subsets
import LinearAlgebra: Symmetric

function MvSim.ρz(D::MvDistribution; n::Int=7)
    d = length(D.margins)
    R = copy(D.R)
    @threads for i in collect(subsets(1:d, Val{2}()))
        R[i...] = ρz(D.R[i...], D.margins[i[1]], D.margins[i[2]], n=n)
    end
    R = Matrix{eltype(R)}(Symmetric(R))
    MvDistribution(R, D.margins, D.C)
end



D = MvDistribution(
    cor_randPD(Float64, 3), 
    [
        Beta(2, 3),
        Binomial(2, 0.2),
        Binomial(20, 0.2)
    ],
    Pearson
)

M = ρz(D; n=18)

x = rand(M, 10000)
D.R - cor(x)
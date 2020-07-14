M = [
    Beta(2, 3),
    Normal(12, 2),
    Exponential(3.14),
    Binomial(1, 0.5),
    LogNormal(0, 1),
    NegativeBinomial(50, 0.01)
]


ρz_bounds(M[1], M[4])
ρz_bounds(M[1], M[5])
ρz_bounds(M[1], M[6])
ρz_bounds(M[2], M[4])
ρz_bounds(M[2], M[5])
ρz_bounds(M[2], M[6])
ρz_bounds(M[3], M[4])
ρz_bounds(M[3], M[5])
ρz_bounds(M[3], M[6])
ρz_bounds(M[4], M[4])
ρz_bounds(M[4], M[5])
ρz_bounds(M[4], M[6])
ρz_bounds(M[5], M[5])
ρz_bounds(M[5], M[6])
ρz_bounds(M[6], M[6])

@time ρz_bounds(M[1], M[1])
@time ρz_bounds(M[1], M[2])
@time ρz_bounds(M[1], M[3])
@time ρz_bounds(M[2], M[2])
@time ρz_bounds(M[2], M[3])
@time ρz_bounds(M[3], M[3])

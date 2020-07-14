using Revise
using MvSim
using Test
using Distributions

dA = Beta(2, 3)
dB = Binomial(2, 0.2)
dC = Binomial(20, 0.2)

# Both Continuous =============================================================
# Values from Table 2
@test -0.914 ≈ ρz(-0.9, dA, dA, 3) atol=0.005
@test -0.611 ≈ ρz(-0.6, dA, dA, 3) atol=0.005
@test -0.306 ≈ ρz(-0.3, dA, dA, 3) atol=0.005
@test  0.304 ≈ ρz( 0.3, dA, dA, 3) atol=0.005
@test  0.606 ≈ ρz( 0.6, dA, dA, 3) atol=0.005
@test  0.904 ≈ ρz( 0.9, dA, dA, 3) atol=0.005

# Both Discrete ===============================================================
# Values from Table 3, Col 1
@test -0.937 ≈ ρz(-0.5, dB, dB, 23) atol=0.005
@test -0.501 ≈ ρz(-0.3, dB, dB,  3) atol=0.005
@test -0.322 ≈ ρz(-0.2, dB, dB,  3) atol=0.005
@test  0.418 ≈ ρz( 0.3, dB, dB,  3) atol=0.005
@test  0.769 ≈ ρz( 0.6, dB, dB,  4) atol=0.005
@test  0.944 ≈ ρz( 0.8, dB, dB, 18) atol=0.005


# Values from Table 3, Col 2
@test -0.939 ≈ ρz(-0.9, dC, dC) atol=0.005
@test -0.624 ≈ ρz(-0.6, dC, dC) atol=0.005
@test -0.311 ≈ ρz(-0.3, dC, dC) atol=0.005
@test  0.310 ≈ ρz( 0.3, dC, dC) atol=0.005
@test  0.618 ≈ ρz( 0.6, dC, dC) atol=0.005
@test  0.925 ≈ ρz( 0.9, dC, dC) atol=0.005

# Mixed =======================================================================
# Values from Table 4, Col 1
@test -0.890 ≈ ρz(-0.7, dB, dA) atol=0.005
@test -0.632 ≈ ρz(-0.5, dB, dA) atol=0.005
@test -0.377 ≈ ρz(-0.3, dB, dA) atol=0.005
@test  0.366 ≈ ρz( 0.3, dB, dA) atol=0.005
@test  0.603 ≈ ρz( 0.5, dB, dA) atol=0.005
@test  0.945 ≈ ρz( 0.8, dB, dA) atol=0.005

# Values from Table 4, Col 2
@test -0.928 ≈ ρz(-0.9, dC, dA) atol=0.005
@test -0.618 ≈ ρz(-0.6, dC, dA) atol=0.005
@test -0.309 ≈ ρz(-0.3, dC, dA) atol=0.005
@test  0.308 ≈ ρz( 0.3, dC, dA) atol=0.005
@test  0.613 ≈ ρz( 0.6, dC, dA) atol=0.005
@test  0.916 ≈ ρz( 0.9, dC, dA) atol=0.005

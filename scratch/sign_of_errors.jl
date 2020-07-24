using Revise
using MvSim
using Test
using Distributions

dA = Beta(2, 3)
dB = Binomial(2, 0.2)
dC = Binomial(20, 0.2)

# Both Continuous =============================================================
# Values from Table 2
-0.914 - ρz(-0.9, dA, dA, 3) > 0
-0.611 - ρz(-0.6, dA, dA, 3) > 0
-0.306 - ρz(-0.3, dA, dA, 3) > 0
0.304 - ρz( 0.3, dA, dA, 3) > 0
0.606 - ρz( 0.6, dA, dA, 3) > 0
0.904 - ρz( 0.9, dA, dA, 3) > 0

# Both Discrete ===============================================================
# Values from Table 3, Col 1
-0.937 - ρz(-0.5, dB, dB, 18) > 0
-0.501 - ρz(-0.3, dB, dB,  3) > 0
-0.322 - ρz(-0.2, dB, dB,  3) > 0
0.418 - ρz( 0.3, dB, dB,  3) > 0
0.769 - ρz( 0.6, dB, dB,  4) > 0
0.944 - ρz( 0.8, dB, dB, 18) > 0


# Values from Table 3, Col 2
-0.939 - ρz(-0.9, dC, dC) > 0
-0.624 - ρz(-0.6, dC, dC) > 0
-0.311 - ρz(-0.3, dC, dC) > 0
0.310 - ρz( 0.3, dC, dC) > 0
0.618 - ρz( 0.6, dC, dC) > 0
0.925 - ρz( 0.9, dC, dC) > 0

# Mixed =======================================================================
# Values from Table 4, Col 1
-0.890 - ρz(-0.7, dB, dA) > 0
-0.632 - ρz(-0.5, dB, dA) > 0
-0.377 - ρz(-0.3, dB, dA) > 0
0.366 - ρz( 0.3, dB, dA) > 0
0.603 - ρz( 0.5, dB, dA) > 0
0.945 - ρz( 0.8, dB, dA) > 0

# Values from Table 4, Col 2
-0.928 - ρz(-0.9, dC, dA) > 0
-0.618 - ρz(-0.6, dC, dA) > 0
-0.309 - ρz(-0.3, dC, dA) > 0
0.308 - ρz( 0.3, dC, dA) > 0
0.613 - ρz( 0.6, dC, dA) > 0
0.916 - ρz( 0.9, dC, dA) > 0

using MvSim
using Test
using Distributions

dA = Beta(2, 3)
dB = Binomial(2, 0.2)
dC = Binomial(20, 0.2)

# Both Continuous =============================================================
# Values from Table 2
-0.914 - ρz(-0.9, dA, dA, n=3)
-0.611 - ρz(-0.6, dA, dA, n=3)
-0.306 - ρz(-0.3, dA, dA, n=3)
 0.304 - ρz( 0.3, dA, dA, n=3)
 0.606 - ρz( 0.6, dA, dA, n=3)
 0.904 - ρz( 0.9, dA, dA, n=3)

# Both Discrete ===============================================================
# Values from Table 3, Col 1
-0.937 - ρz(-0.5, dB, dB, n=18)
-0.501 - ρz(-0.3, dB, dB, n= 3)
-0.322 - ρz(-0.2, dB, dB, n= 3)
 0.418 - ρz( 0.3, dB, dB, n= 3)
 0.769 - ρz( 0.6, dB, dB, n= 4)
 0.944 - ρz( 0.8, dB, dB, n=18)


# Values from Table 3, Col 2
-0.939 - ρz(-0.9, dC, dC)
-0.624 - ρz(-0.6, dC, dC)
-0.311 - ρz(-0.3, dC, dC)
 0.310 - ρz( 0.3, dC, dC)
 0.618 - ρz( 0.6, dC, dC)
 0.925 - ρz( 0.9, dC, dC)

# Mixed =======================================================================
# Values from Table 4, Col 1
-0.890 - ρz(-0.7, dB, dA)
-0.632 - ρz(-0.5, dB, dA)
-0.377 - ρz(-0.3, dB, dA)
 0.366 - ρz( 0.3, dB, dA)
 0.603 - ρz( 0.5, dB, dA)
 0.945 - ρz( 0.8, dB, dA)

# Values from Table 4, Col 2
-0.928 - ρz(-0.9, dC, dA)
-0.618 - ρz(-0.6, dC, dA)
-0.309 - ρz(-0.3, dC, dA)
 0.308 - ρz( 0.3, dC, dA)
 0.613 - ρz( 0.6, dC, dA)
 0.916 - ρz( 0.9, dC, dA)

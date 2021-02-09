using Test
using Bigsimr
using Distributions
import LinearAlgebra: PosDefException
import Bigsimr: ValidCorrelationError

@testset "Random Vector Simulation" begin

    @testset "Random multivariate normal" begin
        r_negdef = [
            1.00 0.82 0.56 0.44
            0.82 1.00 0.28 0.85
            0.56 0.28 1.00 0.22
            0.44 0.85 0.22 1.00
        ]
        # Must fail for negative semidefinite matrices
        @test_throws PosDefException Bigsimr._rmvn(10, r_negdef)
    end

    @testset "rvec" begin
        # Must throw an error if a margin is not a univariate distribution
        r = cor_randPD(2)
        m = [Binomial(10, 0.2), MvNormal(zeros(2), r)]
        @test_throws MethodError rvec(2, r, m)

        # Must throw an error if r is not a valid correlation matrix
        m = [Binomial(10, 0.3), Gamma(10, 3)]
        r = Float64[1.0 2.33333; 0.333333 1.0] # Not positive definite
        c = Float64[2 4; 4 100]                # Is covariance, not correlation
        @test_throws ValidCorrelationError rvec(3, r, m)
        @test_throws ValidCorrelationError rvec(3, c, m)


        # Must throw an arror if the dimensions of r do not match the number of margins
        m = [Binomial(10, 0.3), Gamma(10, 3)]
        r = cor_randPD(3)
        @test_throws DimensionMismatch rvec(4, r, m)
    end

end
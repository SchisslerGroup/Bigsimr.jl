using Test
using Bigsimr
using Distributions
using PDMats

@testset "Utilities" begin

    @testset "Special Functions" begin
        test_types = (Float64, Float32, Float16)
        for T in test_types
            # Must work for each type
            @test_nowarn Bigsimr._normpdf(T(1.0))
            @test_nowarn Bigsimr._normcdf(T(1.0))
            @test_nowarn Bigsimr._norminvcdf(T(0.5))

            # Must respect floating point types
            @test typeof(Bigsimr._normpdf(T(1.0))) === T
            @test typeof(Bigsimr._normcdf(T(1.0))) === T
            @test typeof(Bigsimr._norminvcdf(T(0.5))) === T
        end
    end

    @testset "PDCorMat" begin
        d = 10
        T = Float32
        p = cor_randPD(T, d)
        a = PDCorMat(p, Spearman)

        @test dim(a) == d
        @test size(a) == (d, d)
        @test size(a, 1) == size(a, 2) == d
        @test ndims(a) == 2
        @test eltype(a) == T
        @test Matrix(a) == p
        @test all(diag(a) .== one(T))

        @test_nowarn inv(a)
        @test_nowarn eigmax(a)
        @test_nowarn eigmin(a)
        @test_nowarn logdet(a)

        x = rand(T, d)
        @test_nowarn a * x
        @test_nowarn a \ x

        c = T(π)
        @test_nowarn a * c
        @test_nowarn c * a
        @test a * c == c * a
        @test eltype(a*c) == T


        b = PDCorMat(cor_randPD(Float32, 10), Spearman)
        @test_nowarn a + b
        @test_nowarn pdadd(a, b, c)

        m = rand(T, d, d)
        @test_nowarn pdadd(m, a)        # add `a` to a dense matrix `m` of the same size.
        @test_nowarn pdadd(m, a, c)     # add `a * c` to a dense matrix `m` of the same size.
        @test_nowarn pdadd!(m, a)       # add `a` to a dense matrix `m` of the same size inplace.
        @test_nowarn pdadd!(m, a, c)    # add `a * c` to a dense matrix `m` of the same size inplace.

        r = similar(m)
        @test_nowarn pdadd!(r, m, a)    # add `a` to a dense matrix `m` of the same size, and write the result to `r`.
        @test_nowarn pdadd!(r, m, a, c) # add `a * c` to a dense matrix `m` of the same size, and write the result to `r`.

        @test_nowarn quad(a, x)         # compute x' * a * x when `x` is a vector.
                                        # perform such computation in a column-wise fashion, when
                                        # `x` is a matrix, and return a vector of length `n`,
                                        # where `n` is the number of columns in `x`.
        @test_nowarn invquad(a, x)      # compute x' * inv(a) * x when `x` is a vector.
                                        # perform such computation in a column-wise fashion, when
                                        # `x` is a matrix, and return a vector of length `n`.

        r = similar(x)
        x = rand(T, d, d)
        @test_nowarn quad(a, x)
        @test_nowarn quad!(r, a, x)     # compute x' * a * x in a column-wise fashion, and write the results to `r`.
        @test_nowarn invquad(a, x)      # compute x' * inv(a) * x when `x` is a vector.
                                        # perform such computation in a column-wise fashion, when
                                        # `x` is a matrix, and return a vector of length `n`.
        @test_nowarn invquad!(r, a, x)  # compute x' * inv(a) * x in a column-wise fashion, and write the results to `r`.

        @test_nowarn X_A_Xt(a, x)       # compute `x * a * x'` for a matrix `x`.
        @test_nowarn Xt_A_X(a, x)       # compute `x' * a * x` for a matrix `x`.
        @test_nowarn X_invA_Xt(a, x)    # compute `x * inv(a) * x'` for a matrix `x`.
        @test_nowarn Xt_invA_X(a, x)    # compute `x' * inv(a) * x` for a matrix `x`.

        z = rmvn(1000, p)
        y = similar(z)
        r = similar(z)
        @test_nowarn whiten(a, z)       # whitening transform. `x` can be a vector or a matrix.
        @test_nowarn whiten!(a, y)      # whitening transform inplace, directly updating `y`.
        @test_nowarn whiten!(r, a, z)   # write the transformed result to `r`.

        x = randn(T, 1000, d)
        z = similar(x)
        r = similar(x)
        @test_nowarn unwhiten(a, x)     # inverse of whitening transform. `x` can be a vector or a matrix.
        @test_nowarn unwhiten!(a, z)    # un-whitening transform inplace, updating `z`.
        @test_nowarn unwhiten!(r, a, x) # write the transformed result to `r`.
    end

end

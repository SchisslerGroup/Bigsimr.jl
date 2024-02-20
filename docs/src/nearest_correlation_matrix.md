# Nearest Correlation Matrix

```@setup ncm
using Bigsimr, JLD2, LinearAlgebra, BenchmarkTools
using Statistics, Distributions

brca = JLD2.load("assets/brca200.jld2", "brca")

function fit_mom(x)
    μ = mean(x)
    σ = std(x)
    r = μ^2 / (σ^2 - μ)
    p = μ / σ^2
    NegativeBinomial(r, p)
end

margins = [fit_mom(x) for x in eachcol(brca)]
```

Sometimes what we want really is the Spearman correlation. Then we don't need to do any Pearson matching. All we need to do is estimate/obtain the Spearman correlation of some data, convert it to Pearson, and then simulate. The resulting simulated data will have the same Spearman correlation as the one estimated from the data (up to stochastic error). The problem is that for high dimensional data, the Spearman or converted Pearson correlation matrix may not be positive semidefinite (PSD). The problem is then how to compute the nearest PSD correlation matrix.

We provide the function `cor_nearPD` to handle this problem. It is based off of the work of Qi and Sun (2006), and is a quadratically convergent algorithm. Here we use BRCA data to show its use.

```@repl ncm
m = Matrix(brca)
spearman_corr = cor(m, Spearman);
pearson_corr = cor_convert(spearman_corr, Spearman, Pearson);
isposdef(spearman_corr), isposdef(pearson_corr)
```

We see that the converted Pearson correlation matrix is no longer positve definite. This will result in a failure during the multivariate normal generation, particularly during the Cholesky decomposition.

```@repl ncm
rvec(10, pearson_corr, margins)
```

We can fix this by computing the nearest PD correlation.

```@repl ncm
adjusted_pearson_corr = cor_nearPD(pearson_corr); 
isposdef(adjusted_pearson_corr)
rvec(10, adjusted_pearson_corr, margins)
```

## Benchmarking

What's more impressive is that computing the nearest correlation matrix in Julia is fast!

```julia-repl
julia> @benchmark cor_nearPD(pearson_corr)
BenchmarkTools.Trial: 
  memory estimate:  7.15 MiB
  allocs estimate:  160669
  --------------
  minimum time:     8.968 ms (0.00% GC)
  median time:      9.274 ms (0.00% GC)
  mean time:        9.752 ms (4.78% GC)
  maximum time:     12.407 ms (23.64% GC)
  --------------
  samples:          513
  evals/sample:     1
```

Let's scale up to a larger correlation matrix:

```@repl ncm
m3000 = cor_randPSD(3000) |> m -> cor_convert(m, Spearman, Pearson)
m3000_PD = cor_nearPD(m3000);
isposdef(m3000)
isposdef(m3000_PD)
```

```julia-repl
julia> @benchmark cor_nearPD(m3000)
BenchmarkTools.Trial: 
  memory estimate:  3.95 GiB
  allocs estimate:  78597
  --------------
  minimum time:     10.648 s (2.66% GC)
  median time:      10.648 s (2.66% GC)
  mean time:        10.648 s (2.66% GC)
  maximum time:     10.648 s (2.66% GC)
  --------------
  samples:          1
  evals/sample:     1
```

~11 seconds^[using an AMD Ryzen 9 3900X with 32GB of RAM] to convert a 3000x3000 correlation matrix! This even beats previous benchmarks for a 3000x3000 randomly generated pseudo correlation matrix. Here is an excerpt from Defeng Sun's home page where his matlab code is:

> For a randomly generated  3,000 by 3,000 pseudo correlation matrix (the code is insensitive to input data), the code needs 24 seconds to reach a solution with the relative duality gap less than 1.0e-3 after 3 iterations and 43 seconds  with the relative duality gap less than 1.0e-10 after 6 iterations in my Dell Desktop with Intel (R) Core i7 processor.

We also offer a faster routine that gives up a little accuracy for speed. While [`cor_nearPD`](@ref) finds the nearest correlation matrix to the input matrix, [`cor_fastPD`](@ref) finds a positive definite correlation matrix that is *close* to the input matrix.

```julia-repl
julia> @benchmark cor_fastPD(m3000) samples=10 seconds=30
BenchmarkTools.Trial: 
  memory estimate:  628.92 MiB
  allocs estimate:  26040
  --------------
  minimum time:     1.949 s (0.00% GC)
  median time:      1.998 s (1.15% GC)
  mean time:        2.015 s (2.28% GC)
  maximum time:     2.139 s (8.48% GC)
  --------------
  samples:          10
  evals/sample:     1
```

And it's not too far off from the nearest:

```@repl ncm
m3000_PD_fast = cor_fastPD(m3000);
norm(m3000 - m3000_PD)
norm(m3000 - m3000_PD_fast)
```
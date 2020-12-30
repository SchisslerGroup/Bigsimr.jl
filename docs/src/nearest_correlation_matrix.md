# Nearest Correlation Matrix

```@setup ncm
using MvSim, JLD, LinearAlgebra, BenchmarkTools
using Statistics, Distributions

tmp_dir = mktempdir()
tarball = "assets/brca200.tar.xz"
run(`tar -xf $tarball -C $tmp_dir`)
brca = JLD.load(joinpath(tmp_dir, "brca200.jld"), "brca200")

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
τ = cor(m, Spearman);
ρₚ = cor_convert(τ, Spearman, Pearson);
isposdef.([τ, ρₚ])
```

We see that the converted Pearson correlation matrix is no longer positve definite. This will result in a failure during the multivariate normal generation, particularly during the Cholesky decomposition.

```@repl ncm
rvec(10, ρₚ, margins)
```

We can fix this by computing the nearest PD correlation.

```@repl ncm
ρ̃ₚ = cor_nearPD(ρₚ); 
isposdef(ρ̃ₚ)
rvec(10, ρ̃ₚ, margins)
```

## Benchmarking

What's more impressive is that computing the nearest correlation matrix in Julia is fast!


```@repl ncm
@benchmark cor_nearPD(ρₚ)
```

Let's scale up to a larger correlation matrix:

```@repl ncm
m3000 = cor_randPSD(3000) |> m -> cor_convert(m, Spearman, Pearson)
m3000_PD = cor_nearPD(m3000);
isposdef(m3000)
isposdef(m3000_PD)
@benchmark cor_nearPD(m3000)
```

~3 seconds to convert a 3000x3000 correlation matrix! This even beats previous benchmarks for a 3000x3000 randomly generated pseudo correlation matrix. Here is an excert from Defeng Sun's home page where his matlab code is:

> For a randomly generated  3,000 by 3,000 pseudo correlation matrix (the code is insensitive to input data), the code needs 24 seconds to reach a solution with the relative duality gap less than 1.0e-3 after 3 iterations and 43 seconds  with the relative duality gap less than 1.0e-10 after 6 iterations in my Dell Desktop with Intel (R) Core i7 processor.

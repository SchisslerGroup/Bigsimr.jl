# Pearson Matching

```@setup 1
using MvSim, Distributions
using RDatasets, DataFrames, Statistics
using Plots, PrettyTables
gr()
df = dataset("datasets", "airquality")[:, [:Ozone, :Temp]] |> dropmissing
ρ = cor(Matrix(df), Pearson)
μ_Temp = mean(df.Temp)
σ_Temp = std(df.Temp)
μ_Ozone = mean(log.(df.Ozone))
σ_Ozone = sqrt(mean((log.(df.Ozone) .- mean(log.(df.Ozone))).^2))
margins = [Normal(μ_Temp, σ_Temp), LogNormal(μ_Ozone, σ_Ozone)]
D = MvDistribution(ρ, margins, Pearson);
```

## Correlation Conversion

Let's say we really wanted to estimate the Spearman correlation between the temperature and ozone.

```@repl 1
ρ_s = cor(Matrix(df), Spearman)
cor_bounds(margins[1], margins[2], Spearman)
```

If we just use the Spearman correlation when we simulate data, then the errors are double.

1. The NORTA algorithm is expecting a Pearson correlation
2. The non-linear transformation in the NORTA step does not guarantee that the input correlation is the same as the output.

```@repl 1
D2 = MvDistribution(ρ_s, margins, Spearman);
x_2 = rand(D2, 1_000_000);
cor(x_2, Spearman)
```

Let's try to address **1** and convert the Spearman correlation to a Pearson correlation.

```@repl 1
ρ_p = cor_convert(ρ_s, Spearman, Pearson);
D3 = MvDistribution(ρ_p, margins, Pearson);
x_3 = rand(D3, 1_000_000); 
cor(x_3, Pearson)
cor(x_3, Spearman)
```

Notice that the estimated Pearson correlation is lower than the target Pearson correlation, but the estimated Spearman correlation is essentially the same as the target. This is because the transformation in the NORTA step is monotonic, which means that rank-based correlations are preserved. As a consequence, we can match the Spearman correlation exactly (up to stochastic error), but not the Pearson. 

## Pearson Matching

We can overcome this limitation by employing a Pearson matching algorithm that determines the necessary input correlation in order to achieve the target correlation. Let's now address **2**.

```@repl 1
D4 = pearson_match(D2);
cor(D4)
```

Notice the signficant change in the input correlation!

```@repl 1
x_4 = rand(D4, 1_000_000);
cor(x_4, Pearson)
```

But the estimated correlation is nearly spot on to the [converted] Pearson correlation (ρ_p).

A better example is using the `MvDistribution`. We never estimated the correlation after simulating, so let's look at that now.

```@repl 1
cor(rand(D, 1_000_000))
```

compared to the target correlation:

```@repl 1
ρ
```

The estimated correlation is much too low. Let's do some Pearson matching and observe the results.

```@repl 1
D5 = pearson_match(D); 
x_5 = rand(D5, 1_000_000); 
cor(x_5)
```

Now the simulated data results in a correlation structure that exactly matches the target Pearson correlation!

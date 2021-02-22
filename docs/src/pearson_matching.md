# Pearson Matching

```@setup 1
using Bigsimr, Distributions
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

Here is what we get when we use the Spearman correlation directly with no transformation:

```@repl 1
x_2 = rvec(1_000_000, ρ_s, margins);
cor(x_2, Spearman)
```

Let's try to address **issue 1** and convert the Spearman correlation to a Pearson correlation.

```@repl 1
ρ_p = cor_convert(ρ_s, Spearman, Pearson);
x_3 = rvec(1_000_000, ρ_p, margins);
cor(x_3, Spearman)
```

Notice that the estimated Spearman correlation is essentially the same as the target Spearman correlation. This is because the transformation in the NORTA step is monotonic, which means that rank-based correlations are preserved. As a consequence, we can match the Spearman correlation exactly (up to stochastic error) with an explicit transformation.

## Pearson Matching

We can employ a Pearson matching algorithm that determines the necessary input correlation in order to achieve the target Pearson correlation. Let's now address **issue 2**.

```@repl 1
ρ = cor(Matrix(df), Pearson)
```

If we use the measured correlation directly, then the estimated correlation from the simulated data is far off:

```@repl 1
x_4 = rvec(1_000_000, ρ, margins);
cor(x_4, Pearson)
```

The estimated correlation is much too low. Let's do some Pearson matching and observe the results.

```@repl 1
p = pearson_match(ρ, margins)
x_5 = rvec(1_000_000, p, margins);
cor(x_5)
```

Now the simulated data results in a correlation structure that exactly matches the target Pearson correlation!

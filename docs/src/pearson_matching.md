# Pearson Matching

```@setup 1
using Bigsimr, Distributions
using RDatasets, DataFrames, Statistics
using Plots, PrettyTables
gr()
df = dataset("datasets", "airquality")[:, [:Ozone, :Temp]] |> dropmissing
μ_Temp = mean(df.Temp)
σ_Temp = std(df.Temp)
μ_Ozone = mean(log.(df.Ozone))
σ_Ozone = sqrt(mean((log.(df.Ozone) .- mean(log.(df.Ozone))).^2))
margins = [Normal(μ_Temp, σ_Temp), LogNormal(μ_Ozone, σ_Ozone)]
```

## Correlation Conversion

Let's say we really wanted to estimate the Spearman correlation between the temperature and ozone.

```@repl 1
spearman_corr = cor(Matrix(df), Spearman)
cor_bounds(margins[1], margins[2], Spearman)
```

If we just use the Spearman correlation when we simulate data, then the errors are double.

1. The NORTA algorithm is expecting a Pearson correlation
2. The non-linear transformation in the NORTA step does not guarantee that the input correlation is the same as the output.

Here is what we get when we use the Spearman correlation directly with no transformation:

```@repl 1
x2 = rvec(1_000_000, spearman_corr, margins);
cor(x2, Spearman)
```

Let's try to address **problem 1** and convert the Spearman correlation to a Pearson correlation.

```@repl 1
adjusted_spearman_corr = cor_convert(spearman_corr, Spearman, Pearson);
x3 = rvec(1_000_000, adjusted_spearman_corr, margins);
cor(x3, Spearman)
```

Notice that the estimated Spearman correlation is essentially the same as the target Spearman correlation. This is because the transformation in the NORTA step is monotonic, which means that rank-based correlations are preserved. As a consequence, we can match the Spearman correlation exactly (up to stochastic error) with an explicit transformation.

## Pearson Matching

We can employ a Pearson matching algorithm that determines the necessary input correlation in order to achieve the target Pearson correlation. Let's now address **problem 2**.

```@repl 1
pearson_corr = cor(Matrix(df), Pearson)
```

If we use the measured correlation directly, then the estimated correlation from the simulated data is far off:

```@repl 1
x4 = rvec(1_000_000, pearson_corr, margins);
cor(x4, Pearson)
```

The estimated correlation is much too low. Let's do some Pearson matching and observe the results.

```@repl 1
adjusted_pearson_corr = pearson_match(pearson_corr, margins)
x5 = rvec(1_000_000, adjusted_pearson_corr, margins);
cor(x5)
```

Now the simulated data results in a correlation structure that exactly matches the target Pearson correlation!

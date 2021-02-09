# Getting Started

```@setup started
using Plots, PrettyTables
gr()
```

We’re going to show the basic use and syntax of *bigsimr* by using the New York air quality data set (airquality) included in the *RDatasets* package. We will focus specifically on the temperature (degrees Fahrenheit) and ozone level (parts per billion).

```@example started
using bigsimr, Distributions
using RDatasets, DataFrames, Statistics
```

```@example started
df = dataset("datasets", "airquality")[:, [:Ozone, :Temp]] |> dropmissing
pretty_table(df, tf=tf_markdown, show_row_number=true, vcrop_mode=:middle, display_size=(13, 80)) # hide
```

Let’s look at the joint distribution of the Ozone and Temperature

```@example started
temp_ozone = scatter(df.:Temp, df.:Ozone, legend=false, xguide="Temperature (F)", yguide="Ozone (PPB)") # hide
hist_temp  = histogram(df.:Temp, label="Temperature", bins=30) # hide
hist_ozone = histogram(df.:Ozone, label="Ozone", bins=30) # hide
l = @layout [a{0.7w} grid(2, 1)] # hide
plot(temp_ozone, hist_temp, hist_ozone, layout=l) # hide
```

We can see that not all margins are normally distributed; the ozone level is highly skewed. Though we don’t know the true distribution of ozone levels, we can go forward assuming that it is log-normally distributed.

To simulate observations from this joint distribution, we need to estimate the correlation and the marginal parameters.

## Estimating Correlation

To estimate the correlation, we use `cor` with an argument specifying the type of correlation to estimate. The options are `Pearson`, `Spearman`, or `Kendall`.

```@repl started
ρ = cor(Matrix(df), Pearson)
```

## Defining Marginal Distributions

Next we can estimate the marginal parameters. Assuming that the `Temperature` is normally distributed, it has parameters:

```@repl started
μ_Temp = mean(df.Temp)
σ_Temp = std(df.Temp)
```

and assuming that `Ozone` is log-normally distributed, it has parameters:

```@repl started
μ_Ozone = mean(log.(df.Ozone))
σ_Ozone = sqrt(mean((log.(df.Ozone) .- mean(log.(df.Ozone))).^2))
```

Finally we take the parameters and put them into a vector of margins:

```@repl started
margins = [Normal(μ_Temp, σ_Temp), LogNormal(μ_Ozone, σ_Ozone)]
```

## Multivariate Distribution

While the individual components can be used separately within the package, they work best when joined together into a `MvDistribution` data type:

```@repl started
D = MvDistribution(ρ, margins, Pearson);
```

## Correlation Bounds

Given a vector of margins, the theoretical lower and upper correlation coefficients can be estimated using simulation:

```@repl started
lower, upper = cor_bounds(D);
lower
upper
```

The `pearson_bounds` function uses more sophisticated methods to determine the theoretical lower and upper Pearson correlation bounds. It also requires more computational time.

```@repl started
lower, upper = pearson_bounds(D);
lower
upper
```

## Simulating Multivariate Data

Let’s now simulate 10,000 observations from the joint distribution using `rvec`:

```@repl started
x = rvec(10_000, ρ, margins)
```

Alternatively we can use `rand` with the `MvDistribution` type:

```@repl started
rand(D, 10)
```

## Visualizing Bivariate Data

```@example started
df_sim = DataFrame(x, [:Temp, :Ozone]);

histogram2d(df_sim.:Temp, df_sim.:Ozone, nbins=250, legend=false,
			xlims=extrema(df.:Temp) .+ (-10, 10), 
			ylims=extrema(df.:Ozone) .+ (0, 20))
```

**Compared to Uncorrelated Samples**

We can compare the bivariate distribution above to one where no correlation is taken into account.

```@example started
df_sim2 = DataFrame(
	Temp  = rand(margins[1], 10000), 
	Ozone = rand(margins[2], 10000)
);

histogram2d(df_sim2.:Temp, df_sim2.:Ozone, nbins=250, legend=false,
			xlims=extrema(df.:Temp) .+ (-10, 10), 
			ylims=extrema(df.:Ozone) .+ (0, 20))
```
# Main Functions

## Random Multivariate Vector

### Functions

```@docs
rvec
rmvn
```

## Correlations

### Types

```@docs
Bigsimr.Pearson
Bigsimr.Spearman
Bigsimr.Kendall
```

### Estimating

```@docs
cor
cor_fast
cor_bounds
```

### Generating

```@docs
cor_randPSD
cor_randPD
```

### Converting

Convert a correlation matrix by finding a positive [semi]definite representation.

```@docs
cor_nearPD
cor_fastPD
cor_fastPD!
```

Convert a correlation matrix using other utilities.

```@docs
cor_convert
cov2cor
cov2cor!
cor_constrain
cor_constrain!
```

## Pearson Matching

```@docs
pearson_match
pearson_bounds
```

## Generalized S-Distribution (Experimental)

```@docs
Bigsimr.GSDistribution
```
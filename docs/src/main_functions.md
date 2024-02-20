# API Reference

## Random Multivariate Vector

### Functions

```@docs
rvec
rmvn
```

## Correlations

### Types

```@docs
CorType
Pearson
Spearman
Kendall
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
cor_nearPD!
cor_nearPSD
cor_nearPSD!
cor_fastPD
cor_fastPD!
```

Convert a correlation matrix using other utilities.

```@docs
cor_convert
cor_constrain
cor_constrain!
cov2cor
cov2cor!
```

## Pearson Matching

```@docs
pearson_match
pearson_bounds
```

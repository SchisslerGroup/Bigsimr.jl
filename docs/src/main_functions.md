# API Reference

## Random Vector Generation

```@docs
rvec
rmvn
```

## Correlation Types

```@docs
CorType
Pearson
Spearman
Kendall
```

## Correlation Computation

```@docs
cor
cor_fast
cor_bounds
```

## Random Correlation Matrix

```@docs
cor_randPSD
cor_randPD
```

## Converting Correlation Types

```@docs
cor_convert
```

## Correlation Utils

Convert a correlation matrix using other utilities.

```@docs
cor_constrain
cor_constrain!
cov2cor
cov2cor!
is_correlation
```

## Nearest Correlation Matrix

### Provided by NearestCorrelationMatrix.jl

```@docs
NearestCorrelationMatrix.NearestCorrelationAlgorithm
NearestCorrelationMatrix.Newton
NearestCorrelationMatrix.AlternatingProjection
NearestCorrelationMatrix.DirectProjection
NearestCorrelationMatrix.default_alg
```

```@docs
NearestCorrelationMatrix.nearest_cor
NearestCorrelationMatrix.nearest_cor!
```

### Simplified Wrappers

```@docs
cor_nearPD
cor_nearPD!
cor_nearPSD
cor_nearPSD!
cor_fastPD
cor_fastPD!
```

## Pearson Correlation Matching

```@docs
PearsonCorrelationMatch.pearson_match
PearsonCorrelationMatch.pearson_bounds
```

# Main Functions

## Random Multivariate Vector

```@docs
MvDistribution
rvec
Base.rand(D::MvDistribution, n::Int)
```

## Correlations

```@docs
cor
cor_convert
cor_nearPD(R::Matrix{Float64};
    τ::Float64=1e-5,
    iter_outer::Int=200,
    iter_inner::Int=20,
    N::Int=200,
    err_tol::Float64=1e-6,
    precg_err_tol::Float64=1e-2,
    newton_err_tol::Float64=1e-4)
cor_nearPSD(A::Matrix{T}; n_iter::Int=3) where {T<:Real}
cor_randPSD
```

## Pearson Matching

```@docs
pearson_match(ρx::Real, dA::UnivariateDistribution, dB::UnivariateDistribution; n::Int=7)
pearson_match(D::MvDistribution; n::Int=7)
pearson_bounds
```

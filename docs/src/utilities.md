# Utilities

## General Utilities

```@docs
Bigsimr.hermite(x::Float64, n::Int, probabilists::Bool=true)
Bigsimr.normal_to_margin(d::UnivariateDistribution, x::Float64)
```

## Random Multivariate Vector Utilities

```@docs
Bigsimr._randn(n::Int, d::Int)
Bigsimr._rmvn(n::Int, ρ::Matrix{Float64})
```

## Pearson Matching Utilities

```@docs
Bigsimr.get_coefs(margin::UnivariateDistribution, n::Int)
Bigsimr.Hp(x::Float64, n::Int)
Bigsimr.Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)
Bigsimr.Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UnivariateDistribution , σAσB_inv::Float64)
Bigsimr.solve_poly_pm_one(coef::Vector{Float64})
```

## Nearest Positive Definite Correlation Matrix Utilities

```@docs
Bigsimr.npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
Bigsimr.npd_pca(b::Vector{Float64}, X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
Bigsimr.npd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)
Bigsimr.npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
Bigsimr.npd_set_omega(λ::Vector{Float64}, n::Int)
Bigsimr.npd_jacobian(x::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
```

## Fast Positive Definite Correlation 

```@docs
Bigsimr.fast_pca!(X::Matrix{T}, λ::Vector{T}, P::Matrix{T}, n::Int) where T<:AbstractFloat
```

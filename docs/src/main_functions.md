# Main Functions

## Nearest Correlation Matrix

```@docs
cor_nearPSD(R::Matrix{Float64};
    τ::Float64=1e-5,
    iter_outer::Int=200,
    iter_inner::Int=20,
    N::Int=200,
    err_tol::Float64=1e-6,
    precg_err_tol::Float64=1e-2,
    newton_err_tol::Float64=1e-4)
```

### Utilities

```@docs
MvSim.npsd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
MvSim.npsd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
MvSim.npsd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)
MvSim.npsd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
MvSim.npsd_set_omega(λ::Vector{Float64}, n::Int)
MvSim.npsd_jacobian(x::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int; PERTURBATION::Float64=1e-9)
```

## Pearson Matching

**Continuous**

```@docs
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution, μA, μB, σA, σB, n::Int)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int)
```

**Discrete**

```@docs
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, σA, σB, minA, minB, maxA, maxB, n::Int)
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int)
```

**Mixed**

```@docs
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution, σA, σB, minA, maxA, n::Int)
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution, σA, σB, minB, maxB, n::Int)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int)
```

### Utilities

```@docs
MvSim.get_coefs(::UnivariateDistribution, ::Int)
MvSim.Hϕ(x::T, n::Int) where T<:Real
MvSim.Gn0d(::Int, A, B, α, β, σAσB_inv)
MvSim.Gn0m(::Int, A, α, dB, σAσB_inv)
MvSim.ρz_bounds(::UnivariateDistribution, ::UnivariateDistribution, ::Int)
MvSim.solvePoly_pmOne(coef)
```

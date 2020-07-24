# Main Functions

## Nearest Correlation Matrix

```@docs
cor_nearPSD(R)
```

### Utilities

```@docs
MvSim.npsd_gradient(y, λ₀, P, b₀)
MvSim.npsd_pca(X, λ, P)
MvSim.npsd_pre_cg(b, c, Ω₀, P, ϵ, N)
MvSim.npsd_precond_matrix(Ω₀, P)
MvSim.npsd_set_omega(λ)
MvSim.npsd_jacobian(x, Ω₀, P; PERTURBATION=1e-9)
```

## Pearson Matching

**Continuous**

```@docs
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution, μA, μB, σA, σB, n::Int=3)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int=3)
```

**Discrete**

```@docs
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, σA, σB, minA, minB, maxA, maxB, n::Int=3)
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int=3)
```

**Mixed**

```@docs
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution, σA, σB, minA, maxA, n::Int=3)
ρz(ρx, dA::DiscreteUnivariateDistribution, dB::ContinuousUnivariateDistribution, n::Int=3)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution, σA, σB, minB, maxB, n::Int=3)
ρz(ρx, dA::ContinuousUnivariateDistribution, dB::DiscreteUnivariateDistribution, n::Int=3)
```

### Utilities

```@docs
MvSim.get_coefs(::UnivariateDistribution, ::Int)
MvSim.Gn0d(::Int, A, B, α, β, σAσB_inv)
MvSim.Gn0m(::Int, A, α, dB, σAσB_inv)
MvSim.ρz_bounds(::UnivariateDistribution, ::UnivariateDistribution; ::Int=3)
MvSim.solvePoly_pmOne(coef)
```

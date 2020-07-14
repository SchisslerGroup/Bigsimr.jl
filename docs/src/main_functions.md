# Main Functions

## Nearest Correlation Matrix

```@docs
nearestPSDcor(R)
```

### Utilities

```@docs
MvSim.gradient(y, λ₀, P, b₀)
MvSim.PCA(X, λ, P)
MvSim.pre_cg(b, c, Ω₀, P, ϵ, N)
MvSim.precond_matrix(Ω₀, P)
MvSim.set_omega(λ)
MvSim.jacobian(x, Ω₀, P; PERTURBATION=1e-9)
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

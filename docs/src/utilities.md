# Utilities

```@docs
cor2cor(ρ::T, from::Symbol, to::Symbol) where {T <: Real}
cor2cor(A::Matrix{T}, from::Symbol, to::Symbol) where {T <: Real}
cov2cor!(Σ::Matrix{Float64})
hermite(x, n::Int, probabilists::Bool=true)
rcor(d::Int, α::Real=1.0)
MvSim.setdiag(A::Matrix{T}, x::S) where {T<:Real, S<:Real}
MvSim.z2x(d::UnivariateDistribution, x::AbstractArray)
```

# Utilities

```@docs
cor2cor(::T, ::Symbol, ::Symbol) where {T <: Real}
cor2cor(::AbstractMatrix{T}, ::Symbol, ::Symbol) where {T <: Real}
cov2cor(::AbstractArray)
hermite(x, ::Int, ::Bool=true)
rcor(::Integer, α::Real=1.0)
MvSim.setdiag(A::AbstractMatrix{T}, x::S) where {T<:Real, S<:Real}
MvSim.z2x(d::UnivariateDistribution, x::AbstractArray)
```

"""
    rvec(n, ρ::Matrix, margins::Vector{<:UnivariateDistribution})

Generate samples for a list of marginal distributions and a correaltion structure.
"""
function rvec end
for T in (Float64, Float32, Float16)
    @eval function rvec(n::Int, ρ::Matrix{$T}, margins::Vector{<:UD})
        d = length(margins)
        r,s = size(ρ)

        !(r == s == d) && throw(DimensionMismatch("The number of margins must match the size of the correlation matrix."))
        !iscorrelation(ρ) && throw(ValidCorrelationError())

        Z = SharedMatrix{$T}(_rmvn(n, ρ))
        @inbounds @threads for i in 1:d
            Z[:,i] = normal_to_margin(margins[i], Z[:,i])
        end
        sdata(Z)
    end
    @eval rvec(n::Real, ρ::Matrix{$T}, margins::Vector{<:UD}) = rvec(Int(n), ρ, margins)
end

abstract type CorrelationForm end
struct Adjusted <: CorrelationForm end
struct Target   <: CorrelationForm end
struct General  <: CorrelationForm end

struct CorrelationMatrix{T <: Real, C <: Correlation} <: AbstractMatrix{T}
    x::Matrix{T}
end

const PearsonMatrix{T}  = CorrelationMatrix{T, Pearson}
const SpearmanMatrix{T} = CorrelationMatrix{T, Spearman}
const KendallMatrix{T}  = CorrelationMatrix{T, Kendall}


abstract type CorrelationType end
struct Pearson  <: CorrelationType end
struct Spearman <: CorrelationType end
struct Kendall  <: CorrelationType end

abstract type CorrelationForm end
struct Adjusted <: CorrelationForm end
struct Target   <: CorrelationForm end
struct General  <: CorrelationForm end

abstract type AbstractCorrelation{CT<:CorrelationType, CF<:CorrelationForm} end

struct PearsonCorrelation{T<:Real,CF<:CorrelationForm}  <: AbstractCorrelation{Pearson,  CF}
    r::T
end
struct SpearmanCorrelation{T<:Real,CF<:CorrelationForm} <: AbstractCorrelation{Spearman, CF}
    ρ::T
end
struct KendallCorrelation{T<:Real,CF<:CorrelationForm}  <: AbstractCorrelation{Kendall,  CF}
    τ::T
end

const AdjustedCorrelation{CT<:CorrelationType} = Correlation{CT, Adjusted}
const TargetCorrelation{CT<:CorrelationType}   = Correlation{CT, Target}
const GeneralCorrelation{CT<:CorrelationType}  = Correlation{CT, General}

const AdjustedPearsonCorrelation  = Correlation{Pearson,  Adjusted}
const TargetPearsonCorrelation    = Correlation{Pearson,  Target}
const GeneralPearsonCorrelation   = Correlation{Pearson,  General}
const AdjustedSpearmanCorrelation = Correlation{Spearman, Adjusted}
const TargetSpearmanCorrelation   = Correlation{Spearman, Target}
const GeneralSpearmanCorrelation  = Correlation{Spearman, General}
const AdjustedKendallCorrelation  = Correlation{Kendall,  Adjusted}
const TargetKendallCorrelation    = Correlation{Kendall,  Target}
const GeneralKendallCorrelation   = Correlation{Kendall,  General}

abstract type CorrelationMatrix{T<:AbstractFloat, CT<:CorrelationType, CF<:CorrelationForm} end

const PearsonCorrelationMatrix{T<:AbstractFloat,  CF<:CorrelationForm} = CorrelationMatrix{T, Pearson,  CF}
const KendallCorrelationMatrix{T<:AbstractFloat,  CF<:CorrelationForm} = CorrelationMatrix{T, Kendall,  CF}
const SpearmanCorrelationMatrix{T<:AbstractFloat, CF<:CorrelationForm} = CorrelationMatrix{T, Spearman, CF}

const AdjustedCorrelationMatrix{T<:AbstractFloat, CT<:CorrelationType} = CorrelationMatrix{T, CT, Adjusted}
const TargetCorrelationMatrix{T<:AbstractFloat,   CT<:CorrelationType} = CorrelationMatrix{T, CT, Target}
const GeneralCorrelationMatrix{T<:AbstractFloat,  CT<:CorrelationType} = CorrelationMatrix{T, CT, General}



struct AdjustedPearsonCorrelationMatrix{T<:AbstractFloat} <: CorrelationMatrix{T, Pearson, Adjusted}
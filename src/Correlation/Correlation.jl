abstract type Correlation end
struct Pearson  <: Correlation end
struct Spearman <: Correlation end
struct Kendall  <: Correlation end

export
# Correlation Types
Correlation, Pearson, Spearman, Kendall,
# Correlation Utils
cor_nearPD,
cor_randPSD,
cor_randPD,
cor_convert

include("nearest_pos_def.jl")
include("random.jl")
include("utils.jl")

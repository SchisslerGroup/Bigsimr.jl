export CorType, Pearson, Spearman, Kendall

"""
    CorType

A type used for specifiying the type of correlation. Supported correlations are:

- [`Pearson`](@ref)
- [`Spearman`](@ref)
- [`Kendall`](@ref)
"""
struct CorType{T} end

"""
    Pearson

Pearson's ``r`` product-moment correlation
"""
const Pearson = CorType{:Pearson}()

"""
    Spearman

Spearman's ``ρ`` rank correlation
"""
const Spearman = CorType{:Spearman}()

"""
    Kendall

Kendall's ``τ`` rank correlation
"""
const Kendall = CorType{:Kendall}()

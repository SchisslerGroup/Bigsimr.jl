using Match, Test

struct CorPair
    from::String
    to::String
end

function cor2cor(ρ::T, from::String, to::String) where {T <: Real}
    from, to = lowercase(from), lowercase(to)

    if from == to
        return ρ
    end

    cor_pair = CorPair(from, to)
    @match cor_pair begin
        CorPair("pearson", "spearman") => (6 / π) .* asin.(ρ ./ 2)
        CorPair("pearson", "kendall")  => (2 / π) .* asin.(ρ)
        CorPair("spearman", "pearson") => 2 * sin.(ρ .* π / 6)
        CorPair("spearman", "kendall") => (2 / π) .* asin.(2 * sin.(ρ .* π ./ 6))
        CorPair("kendall", "pearson")  => sin.(ρ .* π / 2)
        CorPair("kendall", "spearman") => (6 / π) .* asin.(sin.(ρ .* π ./ 2) ./ 2)
        CorPair(from, to) => throw(ArgumentError(
            "No matching conversion from '$from' to '$to'. 'from' and 'to' must be any combination of 'pearson', 'spearman', or 'kendall'"
        ))
    end
end

cor2cor(0.5, "pearson", "pearson")
cor2cor(0.5, "pearson", "spearman")
cor2cor(0.5, "pearson", "kendall")
cor2cor(0.5, "spearman", "pearson")
cor2cor(0.5, "spearman", "spearman")
cor2cor(0.5, "spearman", "kendall")
cor2cor(0.5, "kendall", "pearson")
cor2cor(0.5, "kendall", "spearman")
cor2cor(0.5, "kendall", "kendall")

cor2cor(0.5, "notavalid", "typeofcorr")

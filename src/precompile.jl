precompile(rmvn, (Int, Vector{Float64}, Matrix{Float64}))
precompile(rmvn, (Float64, Vector{Float64}, Matrix{Float64}))
precompile(rmvn, (Int, Matrix{Float64}))
precompile(rmvn, (Float64, Matrix{Float64}))

precompile(cor, (Matrix{Float64}, Pearson))
precompile(cor, (Matrix{Float64}, Spearman))
precompile(cor, (Matrix{Float64}, Kendall))

precompile(cor_fast, (Matrix{Float64}, Pearson))
precompile(cor_fast, (Matrix{Float64}, Spearman))
precompile(cor_fast, (Matrix{Float64}, Kendall))

precompile(cor_randPD, (Float64, Int, Int))
precompile(cor_randPSD, (Float64, Int, Int))

precompile(cor_fastPD!, (Matrix{Float64}, Float64))
precompile(cor_fastPD, (Matrix{Float64}, Float64))

precompile(cor_nearPD, (Matrix{Float64},))
precompile(cor_nearPD, (Matrix{Float64}, Float64, Float64))
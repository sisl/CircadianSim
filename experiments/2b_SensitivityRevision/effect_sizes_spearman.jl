# Reviewer comment 2.6: report effect sizes rather than significance for the Fig 2 correlations.
# On the revision baseline (967 particles, same in both conditions):
#   - Spearman ρ for inputs × outputs (30) and output pairs (15), per light condition
#   - 95 % percentile bootstrap CIs (particles resampled with replacement)
#   - p-values (t approximation) with Holm correction over the 45 tests of each condition
# Output: results/spearman_effect_sizes.csv (ρ, bootstrap CI, raw and Holm-adjusted p for every pair).
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/effect_sizes_spearman.jl

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using Random
using StableRNGs
using StatsBase
using Distributions
using DataFrames
using CSV
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const N_BOOT = 2000
const SEED = 42
const KEEP = analysis_inputs(load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"])   # ψ drops out when fixed at 0
const INPUTS = ALL_INPUTS[KEEP]
const OUTPUTS = [:s_on, :s_off, :s_dur, :s_mp, :cbtmin, :psi_obs]
const INPUT_LABELS = ALL_INPUT_LABELS[KEEP]
const OUTPUT_LABELS = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"]
const ALPHA = 0.05

"""Holm step-down adjusted p-values."""
function holm(p)
    m = length(p); o = sortperm(p); adj = similar(p, Float64); running = 0.0
    for (rank, i) in enumerate(o)
        running = max(running, min(1.0, (m - rank + 1) * p[i]))
        adj[i] = running
    end
    return adj
end

spearman_p(ρ, n) = (t = ρ * sqrt((n - 2) / max(1 - ρ^2, eps())); 2 * ccdf(TDist(n - 2), abs(t)))

baseline = load(joinpath(RESULTS_DIR, "deterministic_baseline_summaries.jld2"))["baseline"]
println("Revision baseline: $(length(baseline.included_ids)) particles in every condition"); flush(stdout)

rows = NamedTuple[]
for condition in REGULAR_SCHEDULE_CONDITIONS
    f = extract_summary_features(baseline.by_condition[condition].analysis; normalize_time_outputs=true)
    X = hcat([getproperty(f.inputs, k) for k in INPUTS]...)
    Y = hcat([getproperty(f.outputs, k) for k in OUTPUTS]...)
    all(isfinite, X) && all(isfinite, Y) || error("non-finite features in $condition")
    n = size(X, 1)

    pairs = vcat([(:io, i, j) for i in eachindex(INPUTS) for j in eachindex(OUTPUTS)],
                 [(:oo, i, j) for i in eachindex(OUTPUTS) for j in eachindex(OUTPUTS) if i < j])
    col(kind, i) = kind == :io ? X[:, i] : Y[:, i]
    ρ = [corspearman(col(k, i), Y[:, j]) for (k, i, j) in pairs]

    rng = StableRNG(SEED)
    boot = Matrix{Float64}(undef, N_BOOT, length(pairs))
    for b in 1:N_BOOT
        idx = rand(rng, 1:n, n)
        for (c, (k, i, j)) in enumerate(pairs)
            boot[b, c] = corspearman(col(k, i)[idx], Y[idx, j])
        end
    end
    lo = [quantile(boot[:, c], 0.025) for c in eachindex(pairs)]
    hi = [quantile(boot[:, c], 0.975) for c in eachindex(pairs)]
    p = spearman_p.(ρ, n)
    p_holm = holm(p)

    for (c, (k, i, j)) in enumerate(pairs)
        push!(rows, (condition=condition, n=n, kind=k == :io ? "input-output" : "output-output",
                     var1=k == :io ? INPUT_LABELS[i] : OUTPUT_LABELS[i], var2=OUTPUT_LABELS[j],
                     rho=ρ[c], ci_lo=lo[c], ci_hi=hi[c], ci_includes_zero=lo[c] <= 0 <= hi[c],
                     p=p[c], p_holm=p_holm[c]))
    end

    d = DataFrame(rows[end-length(pairs)+1:end])
    println("\n$condition (n = $n):")
    println("  |ρ| ≥ 0.5: $(count(>=(0.5), abs.(d.rho)))   0.3–0.5: $(count(x -> 0.3 <= x < 0.5, abs.(d.rho)))   " *
            "0.1–0.3: $(count(x -> 0.1 <= x < 0.3, abs.(d.rho)))   < 0.1: $(count(<(0.1), abs.(d.rho)))  (of $(nrow(d)))")
    println("  CI includes 0: $(count(d.ci_includes_zero));  p < 0.05 raw: $(count(<(0.05), d.p)), Holm: $(count(<(0.05), d.p_holm))")
    println("  Smallest |ρ| with Holm p < 0.05: $(round(minimum(abs.(d.rho[d.p_holm .< 0.05]); init=Inf), digits=3))")
    flush(stdout)

end

max_hw = maximum((r.ci_hi - r.ci_lo) / 2 for r in rows)
println("\nAll 95 % bootstrap CIs within ±$(@sprintf("%.2f", max_hw)) (n = $(first(rows).n)).")
CSV.write(joinpath(RESULTS_DIR, "spearman_effect_sizes.csv"), DataFrame(rows))
println("\nWrote results/spearman_effect_sizes.csv")

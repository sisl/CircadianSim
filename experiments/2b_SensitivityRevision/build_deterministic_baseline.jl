# Revision baseline: 70-day regular schedule (stochastic light; the file name is from an earlier deterministic-light
# version) on experiment 2's 1,000 initial
# particles, MLE summaries over days 56–70, entrainment rule from sensitivity_common.jl.
# Saves the baseline for the 2.6 / 3.6 / 2.4 analyses and reports:
#   - how many particles are excluded and why (not asleep every night vs outside the ±1 h band); a particle
#     failing in either condition is excluded from every analysis
#   - Spearman ρ and mean |SHAP| for: 70-day included (main), 70-day all, and the 35-day window
#     (experiment 2's days 20.3–35, same run) to show what the burn-in change moves.
# SHAP uses computeSHAP.jl's settings (100 trees, random_state=42, in-sample TreeExplainer).
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/build_deterministic_baseline.jl

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using StatsBase
using DataFrames
using CSV
using PyCall

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const INPUTS = ALL_INPUTS[analysis_inputs(load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"])]   # ψ drops out when fixed at 0
const OUTPUTS = [:s_on, :s_off, :s_dur, :s_mp, :cbtmin, :psi_obs]

"""Finite input/output feature matrices (experiment-2 normalization) for a set of summaries."""
function feature_matrices(summaries)
    f = extract_summary_features(summaries; normalize_time_outputs=true)
    X = hcat([getproperty(f.inputs, k) for k in INPUTS]...)
    Y = hcat([getproperty(f.outputs, k) for k in OUTPUTS]...)
    ok = vec(all(isfinite, hcat(X, Y); dims=2))
    return X[ok, :], Y[ok, :]
end

spearman_matrix(X, Y) = [corspearman(X[:, i], Y[:, j]) for i in axes(X, 2), j in axes(Y, 2)]

py"""
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

def mean_abs_shap(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)
"""
shap_matrix(X, Y) = hcat([py"mean_abs_shap"(X, Y[:, j]) for j in axes(Y, 2)]...)

importance_ranks(M) = hcat([invperm(sortperm(abs.(M[:, j]); rev=true)) for j in axes(M, 2)]...)
top_inputs(M) = [INPUTS[argmax(abs.(M[:, j]))] for j in axes(M, 2)]

"""Events truncated to before `days` (to score an earlier window from a longer run)."""
truncate_events(events, days) = map(ev -> filter(e -> e.t_hours < 24days, ev), events)

particles = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]
println("Loaded $(length(particles)) initial particles; threads = $(Threads.nthreads())")
println("Baseline: $(REVISION_TOTAL_DAYS) days, scored on days $(REVISION_SCORE_START_DAY)–$(REVISION_TOTAL_DAYS), " *
        "entrainment band ±$(ENTRAINMENT_BAND_H) h"); flush(stdout)

baseline = build_revision_baseline(particles)
included = Set(baseline.included_ids)
println("\nIncluded in every analysis (entrained with a summary in all conditions): $(length(included)) of $(length(particles))")

rows = NamedTuple[]
for condition in REGULAR_SCHEDULE_CONDITIONS
    println("\n", "="^72, "\n$condition\n", "="^72); flush(stdout)
    b = baseline.by_condition[condition]
    metrics = b.metrics
    no_sleep = Set(m.particle_id for m in metrics if !(m.min_night_sleep_fraction > NIGHT_SLEEP_MIN_FRACTION))
    off_band = Set(m.particle_id for m in metrics if m.min_night_sleep_fraction > NIGHT_SLEEP_MIN_FRACTION && !m.entrained)
    no_summary = setdiff(Set(1:length(particles)), Set(s.particle_id for s in b.summaries))
    fails_here = union(no_sleep, off_band, no_summary)
    println("Fails the rule in this condition: $(length(fails_here)) " *
            "= $(length(no_sleep)) not asleep every night + $(length(off_band)) sleeping every night but outside the band" *
            (isempty(setdiff(no_summary, no_sleep)) ? "" : " + $(length(setdiff(no_summary, no_sleep))) without a summary"))
    println("  Passes here but excluded because it fails in the other condition: $(length(setdiff(setdiff(Set(1:length(particles)), fails_here), included)))")
    println("Analysis population: $(length(b.analysis))"); flush(stdout)

    events35 = truncate_events(simulate_regular_schedule_events(particles, condition), LEGACY_WINDOW.total_days)
    sets = ["70d_included" => b.analysis, "70d_all" => b.summaries,
            "35d_all" => summarize_window(events35, particles; score_start_day=LEGACY_WINDOW.score_start_day)]
    R, S = Dict{String,Matrix{Float64}}(), Dict{String,Matrix{Float64}}()
    for (name, ss) in sets
        X, Y = feature_matrices(ss)
        R[name], S[name] = spearman_matrix(X, Y), shap_matrix(X, Y)
        println("  $name: n = $(size(X, 1))")
        for (metric, M) in (("spearman", R[name]), ("mean_abs_shap", S[name])), a in eachindex(INPUTS), b_ in eachindex(OUTPUTS)
            push!(rows, (condition=condition, subset=name, n=size(X, 1), metric=metric,
                         input=String(INPUTS[a]), output=String(OUTPUTS[b_]), value=M[a, b_]))
        end
    end
    for other in ("70d_all", "35d_all")
        println("\n70d_included vs $other:")
        println("  max |Δρ| = $(round(maximum(abs.(R[other] .- R["70d_included"])), digits=3))")
        println("  importance-rank changes of 30: Spearman $(sum(importance_ranks(R[other]) .!= importance_ranks(R["70d_included"]))), " *
                "SHAP $(sum(importance_ranks(S[other]) .!= importance_ranks(S["70d_included"])))")
        for (j, o) in enumerate(OUTPUTS)
            tr, to = top_inputs(R["70d_included"])[j], top_inputs(R[other])[j]
            sr, so = top_inputs(S["70d_included"])[j], top_inputs(S[other])[j]
            (tr != to || sr != so) && println("  $o top input changes: ρ $to → $tr, SHAP $so → $sr ($other → 70d_included)")
        end
    end
    flush(stdout)
end

CSV.write(joinpath(RESULTS_DIR, "baseline_matrices_70d_vs_35d.csv"), DataFrame(rows))
@save joinpath(RESULTS_DIR, "deterministic_baseline_summaries.jld2") baseline
println("\nSaved results/deterministic_baseline_summaries.jld2 and results/baseline_matrices_70d_vs_35d.csv")

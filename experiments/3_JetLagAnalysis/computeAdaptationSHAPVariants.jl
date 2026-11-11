# Fig 4C–D feature-set variants: SHAP of jet-lag adaptation time on
#   "parameters" : static parameters only (τ, k, ψ, χ, μ_H/χ)                        — main text
#   "outputs"    : regular-schedule steady-state timing only (CBTmin, S_On, S_Off, ψ_obs, S_Dur)
#   "all"        : both (the manuscript's original feature set; written by summarizeAdaptationStatistics.jl)
# The outputs are functions of the parameters, so in "all" the forest credits the outputs and the parameters
# look unimportant; the separate variants show each feature set's own attribution.
#
# Per light condition and shift: features from each included particle's own summary, joined to its
# adaptation time by particle_id (as in the #27 fix); particles not adapted within the post-shift window enter at the
# window length (censored), as in summarizeAdaptationStatistics.jl; RandomForestRegressor(100 trees, random_state=42) and
# TreeExplainer, as elsewhere. Also reports out-of-fold R² (5-fold × 3 repeats) for every feature set,
# i.e. how much of adaptation time each explains.
#
# Writes jetlag_shap_cache_path(condition; variant, target) for "parameters", "outputs" and "all" (read by
# `plotCombinedLightConditions.jl <variant>`) and a CSV of the R² comparison. Each cache holds mean |SHAP| on the
# fitted scale ("importance_by_tz") and, for presentation, on the day scale ("importance_days_by_tz"): with the
# log(1 + days) target each participant's contribution is converted to first order, |SHAP| × (1 + predicted days)
# (d days / d log(1 + days) = 1 + days), then averaged; with the raw-day target the two are the same.
# Run: julia -t auto --project=. experiments/3_JetLagAnalysis/computeAdaptationSHAPVariants.jl

using CircadianPOMDP
using JLD2
using Statistics
using DataFrames
using CSV
using PyCall
using Printf

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

py"""
import numpy as np, shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

def rf():
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

def mean_abs_shap(X, y, log_target):
    m = rf().fit(X, y)
    sv = np.abs(shap.TreeExplainer(m).shap_values(X))
    days = (sv * np.exp(m.predict(X))[:, None]).mean(axis=0) if log_target else sv.mean(axis=0)
    return sv.mean(axis=0), days

def oof_r2(X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    return float(np.mean(cross_val_score(rf(), X, y, cv=cv, scoring="r2")))
"""

const FEATURES = ["τ", "k", "ψ", "χ", "μ_H/χ", "CBTmin", "Sleep Onset", "Sleep Offset", "Phase Offset", "Sleep Duration"]
# "all" matches summarizeAdaptationStatistics.jl's own fit (same features, forest and seed), so either can write it
const VARIANT_COLUMNS = Dict("parameters" => 1:5, "outputs" => 6:10, "all" => 1:10)

r2_rows = NamedTuple[]
for condition in REGULAR_SCHEDULE_CONDITIONS
    res = load(steady_state_results_path(condition=condition))
    shifts, window = res["timezone_shifts"], Float64(res["post_shift_window_days"])
    ids_all = [vcat(a, n) for (a, n) in zip(res["steady_state_particle_ids_by_timezone"], res["not_adapted_particle_ids_by_timezone"])]
    days_all = [vcat(Float64.(a), fill(window, length(n)))
                for (a, n) in zip(res["steady_state_days_by_timezone"], res["not_adapted_particle_ids_by_timezone"])]
    summaries = filter_included(load_particle_summaries(regular_summaries_path(; condition)))
    f = extract_summary_features(summaries; normalize_time_outputs=true)
    rows = Dict(s.particle_id => [f.inputs.tau[k], f.inputs.k[k], f.inputs.psi[k], f.inputs.chi[k], f.inputs.mu_h_over_chi[k],
                                  f.outputs.cbtmin[k], f.outputs.s_on[k], f.outputs.s_off[k], f.outputs.psi_obs[k], f.outputs.s_dur[k]]
                for (k, s) in enumerate(summaries))
    importance = Dict(v => Dict{Int,Vector{Float64}}() for v in keys(VARIANT_COLUMNS))
    importance_days = Dict(v => Dict{Int,Vector{Float64}}() for v in keys(VARIANT_COLUMNS))
    println("\n$condition  (shift: n, out-of-fold R² parameters | outputs | all; top parameter, top output)")
    for (k, tz) in enumerate(shifts)
        keep = [i for i in eachindex(ids_all[k]) if haskey(rows, ids_all[k][i]) && all(isfinite, rows[ids_all[k][i]])]
        length(keep) < 10 && continue
        X = permutedims(hcat([rows[ids_all[k][i]] for i in keep]...))
        y = adaptation_shap_y(days_all[k][keep])   # log(1 + days) unless SHAP_TARGET=days
        for (v, cols) in VARIANT_COLUMNS
            importance[v][tz], importance_days[v][tz] = py"mean_abs_shap"(X[:, cols], y, ADAPTATION_SHAP_TARGET == "logdays")
        end
        r2p, r2o, r2a = py"oof_r2"(X[:, 1:5], y), py"oof_r2"(X[:, 6:10], y), py"oof_r2"(X, y)
        push!(r2_rows, (condition=condition, tz=tz, n=length(keep), r2_parameters=r2p, r2_outputs=r2o, r2_all_features=r2a))
        tp = FEATURES[1:5][argmax(importance["parameters"][tz])]
        to = FEATURES[6:10][argmax(importance["outputs"][tz])]
        println(@sprintf("  %2d h: n=%d  R² %.2f | %.2f | %.2f  top %s, %s", tz, length(keep), r2p, r2o, r2a, tp, to))
        flush(stdout)
    end
    for v in keys(VARIANT_COLUMNS)
        out = jetlag_shap_cache_path(condition=condition, variant=v)
        jldopen(out, "w") do file
            file["importance_by_tz"] = importance[v]
            file["importance_days_by_tz"] = importance_days[v]
        end
        println("  Saved $out")
    end
end
csv = adaptation_r2_csv_path()
CSV.write(csv, DataFrame(r2_rows))
println("\nSaved $csv")

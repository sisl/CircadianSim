# Reviewer comment 3.6: how well do the Random Forest surrogates fit, and how stable are the SHAP
# rankings? On the revision baseline (967 particles, same in both conditions), per light condition and
# output, with computeSHAP.jl's model (RandomForestRegressor, 100 trees):
#   - Fit: out-of-fold R² from 5-fold CV repeated 10× (50 fits), a linear-regression R² from the same
#     folds for comparison, and the in-sample R² of the model the paper explains.
#   - SHAP stability:
#       seeds       : 10 RFs on all data (random_state 0–9), path-dependent TreeExplainer as in the paper
#       folds       : SHAP of each CV model on its held-out fold (50 estimates)
#       interventional : random_state=42 model, interventional TreeExplainer with a 200-particle
#                     background (does not split credit through the χ – μ_H/χ correlation)
#     Rank agreement across the 10 seeds (Kendall's W; share of seeds with the paper's top input).
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/rf_fit_and_shap_stability.jl

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using StatsBase
using DataFrames
using CSV
using Plots
using PyCall
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")
const FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(RESULTS_DIR); mkpath(FIG_DIR)

const KEEP = analysis_inputs(load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"])   # ψ drops out when fixed at 0
const INPUTS = ALL_INPUTS[KEEP]
const OUTPUTS = [:s_on, :s_off, :s_dur, :s_mp, :cbtmin, :psi_obs]
const INPUT_LABELS = ALL_INPUT_LABELS[KEEP]
const OUTPUT_LABELS = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"]

py"""
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

def rf(seed):
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

def analyse(X, y, n_splits=5, n_repeats=10, n_seeds=10, n_background=200):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    out = {"r2_rf_oof": [], "r2_linear_oof": [], "shap_folds": []}
    for k, (tr, te) in enumerate(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42).split(X)):
        m = rf(k).fit(X[tr], y[tr])
        out["r2_rf_oof"].append(r2_score(y[te], m.predict(X[te])))
        lin = LinearRegression().fit(X[tr], y[tr])
        out["r2_linear_oof"].append(r2_score(y[te], lin.predict(X[te])))
        out["shap_folds"].append(np.abs(shap.TreeExplainer(m).shap_values(X[te])).mean(axis=0))
    out["shap_seeds"] = []
    for s in range(n_seeds):
        m = rf(s).fit(X, y)
        out["shap_seeds"].append(np.abs(shap.TreeExplainer(m).shap_values(X)).mean(axis=0))
    m42 = rf(42).fit(X, y)
    out["r2_rf_insample"] = r2_score(y, m42.predict(X))
    out["shap_paper"] = np.abs(shap.TreeExplainer(m42).shap_values(X)).mean(axis=0)
    bg = X[np.random.default_rng(42).choice(len(X), size=min(n_background, len(X)), replace=False)]
    iexp = shap.TreeExplainer(m42, data=bg, feature_perturbation="interventional")
    out["shap_interventional"] = np.abs(iexp.shap_values(X, check_additivity=False)).mean(axis=0)
    for key in ("shap_folds", "shap_seeds"):
        out[key] = np.vstack(out[key])
    return out
"""

"""Kendall's W for an m × n matrix of scores (m raters ranking n items; larger score = rank 1)."""
function kendalls_w(S)
    m, n = size(S)
    R = hcat([invperm(sortperm(S[r, :]; rev=true)) for r in 1:m]...)'   # m × n ranks
    Rsum = vec(sum(R; dims=1))
    return 12 * sum((Rsum .- mean(Rsum)) .^ 2) / (m^2 * (n^3 - n))
end

baseline = load(joinpath(RESULTS_DIR, "deterministic_baseline_summaries.jld2"))["baseline"]
println("Revision baseline: $(length(baseline.included_ids)) particles in every condition"); flush(stdout)

fit_rows, shap_rows, rank_rows = NamedTuple[], NamedTuple[], NamedTuple[]
for condition in REGULAR_SCHEDULE_CONDITIONS
    f = extract_summary_features(baseline.by_condition[condition].analysis; normalize_time_outputs=true)
    X = hcat([getproperty(f.inputs, k) for k in INPUTS]...)
    println("\n", "="^72, "\n$condition (n = $(size(X, 1)))\n", "="^72)
    ic, im = findfirst(==(:chi), INPUTS), findfirst(==(:mu_h_over_chi), INPUTS)
    println("  corr(χ, μ_H/χ): Spearman $(round(corspearman(X[:, ic], X[:, im]), digits=3)), " *
            "Pearson $(round(cor(X[:, ic], X[:, im]), digits=3))")
    for (j, o) in enumerate(OUTPUTS)
        y = getproperty(f.outputs, o)
        t = @elapsed r = py"analyse"(X, y)
        oof, lin = r["r2_rf_oof"], r["r2_linear_oof"]
        push!(fit_rows, (condition=condition, output=OUTPUT_LABELS[j],
                         rf_oof_mean=mean(oof), rf_oof_sd=std(oof), rf_oof_min=minimum(oof),
                         linear_oof_mean=mean(lin), linear_oof_sd=std(lin), rf_insample=r["r2_rf_insample"]))
        for (src, M) in (("seeds", r["shap_seeds"]), ("folds", r["shap_folds"]))
            for (i, lab) in enumerate(INPUT_LABELS)
                push!(shap_rows, (condition=condition, output=OUTPUT_LABELS[j], input=lab, source=src,
                                  mean=mean(M[:, i]), sd=std(M[:, i]), min=minimum(M[:, i]), max=maximum(M[:, i])))
            end
        end
        for (src, v) in (("paper_seed42", r["shap_paper"]), ("interventional", r["shap_interventional"]))
            for (i, lab) in enumerate(INPUT_LABELS)
                push!(shap_rows, (condition=condition, output=OUTPUT_LABELS[j], input=lab, source=src,
                                  mean=v[i], sd=0.0, min=v[i], max=v[i]))
            end
        end
        paper_rank = invperm(sortperm(r["shap_paper"]; rev=true))
        seed_ranks = [invperm(sortperm(r["shap_seeds"][s, :]; rev=true)) for s in axes(r["shap_seeds"], 1)]
        fold_ranks = [invperm(sortperm(r["shap_folds"][s, :]; rev=true)) for s in axes(r["shap_folds"], 1)]
        top = argmin(paper_rank)
        int_rank = invperm(sortperm(r["shap_interventional"]; rev=true))
        push!(rank_rows, (condition=condition, output=OUTPUT_LABELS[j], paper_top_input=INPUT_LABELS[top],
                          kendalls_w_seeds=kendalls_w(r["shap_seeds"]), kendalls_w_folds=kendalls_w(r["shap_folds"]),
                          share_seeds_same_top=mean(sr[top] == 1 for sr in seed_ranks),
                          share_folds_same_top=mean(fr[top] == 1 for fr in fold_ranks),
                          share_seeds_same_full_ranking=mean(sr == paper_rank for sr in seed_ranks),
                          interventional_top_input=INPUT_LABELS[argmin(int_rank)],
                          interventional_same_ranking=int_rank == paper_rank))
        fr = fit_rows[end]; rr = rank_rows[end]
        println(@sprintf("  %-8s R² RF OOF %.3f ± %.3f (min %.3f) | linear OOF %.3f | RF in-sample %.3f | top %s: seeds %.0f%%, folds %.0f%%, W_seeds %.2f | interventional top %s  (%.0f s)",
                         OUTPUT_LABELS[j], fr.rf_oof_mean, fr.rf_oof_sd, fr.rf_oof_min, fr.linear_oof_mean, fr.rf_insample,
                         rr.paper_top_input, 100rr.share_seeds_same_top, 100rr.share_folds_same_top, rr.kendalls_w_seeds,
                         rr.interventional_top_input, t))
        flush(stdout)
    end
end

fit_df, shap_df, rank_df = DataFrame(fit_rows), DataFrame(shap_rows), DataFrame(rank_rows)
CSV.write(joinpath(RESULTS_DIR, "rf_cv_r2.csv"), fit_df)
CSV.write(joinpath(RESULTS_DIR, "shap_stability.csv"), shap_df)
CSV.write(joinpath(RESULTS_DIR, "shap_rank_agreement.csv"), rank_df)

# Figures (drawn at print size from the CSVs above)
include(joinpath(@__DIR__, "plot_rf_fit_and_shap_stability.jl"))
println("\nWrote results/rf_cv_r2.csv, shap_stability.csv, shap_rank_agreement.csv; figures/rf_cv_r2.png, shap_stability.png")

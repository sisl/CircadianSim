# Reviewer comment 2.4: do the Spearman and SHAP conclusions depend on the parameter SDs?
#
# Each scenario rescales the spread of one static parameter (or all of them) and re-runs the revision
# baseline pipeline. Common random numbers: every scenario starts from experiment 2's 1,000 initial
# particles (same initial states) and keeps each particle's standardized draw
#   z = (x − mean) / SD,  x_scenario = mean + scale · SD · z,
# so scale 1 reproduces the baseline exactly and differences between scenarios come only from the
# SD change. The uniform scenario maps z → U(−√3, √3) (same mean and SD, same particle ranks).
# ψ is fixed at 0 in the baseline; the φ scenarios add a between-person phase lag ψ = SD · z_ψ, with z_ψ the
# standard-normal draws of the particles' own generation (psi_draws), shared by every φ scenario.
#
# Per scenario: entrainment exclusions (same rule as the baseline, both conditions), Spearman ρ with
# 95 % bootstrap CIs, and mean |SHAP| (computeSHAP.jl settings, random_state=42) for each condition.
# Scenarios are checkpointed to results/sd_sweep/ and skipped if already present.
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/sd_sweep.jl

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using Random
using StableRNGs
using StatsBase
using Distributions
using DataFrames
using CSV
using PyCall

const RESULTS_DIR = joinpath(@__DIR__, "results")
const SWEEP_DIR = joinpath(RESULTS_DIR, "sd_sweep")
mkpath(SWEEP_DIR)

const N_BOOT = 500
const SEED = 42

# Prior means and SDs used to generate experiment 2's particles (initialize_particles defaults; ψ fixed at 0)
const PRIOR = (tau_x=(24.2, 0.2), K=(0.55, 0.05), chi=(45.0, 5.0), mu_h=(4.2, 0.5))
const PARAMS = keys(PRIOR)

const SCALES = (0.5, 1.5, 2.0)
const PHI_SD_H = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)   # ψ SDs added to the ψ = 0 baseline (~1 h: between-person SD of the observed phase angle)

const OUTPUTS = [:s_on, :s_off, :s_dur, :s_mp, :cbtmin, :psi_obs]

"""Scenarios: (name, varied, scale per parameter and absolute ψ SD in h as :phi, distribution)."""
function scenarios()
    one = merge(Dict(p => 1.0 for p in PARAMS), Dict(:phi => 0.0))
    list = [("baseline", "none", one, :normal)]
    for p in (:tau_x, :K, :chi, :mu_h), s in SCALES
        push!(list, ("$(p)_x$(s)", String(p), merge(one, Dict(p => s)), :normal))
    end
    for sd in PHI_SD_H
        push!(list, ("phi_sd$(sd)h", "phi", merge(one, Dict(:phi => sd)), :normal))
    end
    for s in (0.5, 2.0)
        push!(list, ("all_x$(s)", "all", merge(Dict(p => s for p in PARAMS), Dict(:phi => 0.0)), :normal))
    end
    push!(list, ("uniform", "shape", one, :uniform))
    return list
end

"""Particles with rescaled static parameters and ψ = scale[:phi] · z_ψ (initial states unchanged)."""
function rescale_particles(particles, scale, dist, z_psi)
    to_u(z) = sqrt(3) * (2 * cdf(Normal(), z) - 1)   # rank-preserving map to U(−√3, √3), SD 1
    map(enumerate(particles)) do (i, p)
        vals = Dict(f => getfield(p, f) for f in fieldnames(CircadianState))
        for q in PARAMS
            dist == :normal && scale[q] == 1.0 && continue   # keep the original value bit-for-bit
            m, sd = PRIOR[q]
            z = (getfield(p, q) - m) / sd
            vals[q] = m + scale[q] * sd * (dist == :uniform ? to_u(z) : z)
        end
        scale[:phi] > 0 && (vals[:phi] = scale[:phi] * z_psi[i])
        CircadianState((vals[f] for f in fieldnames(CircadianState))...)
    end
end

py"""
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

def mean_abs_shap(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)
"""

function analyse_condition(summaries, inputs)
    f = extract_summary_features(summaries; normalize_time_outputs=true)
    X = hcat([getproperty(f.inputs, k) for k in inputs]...)
    Y = hcat([getproperty(f.outputs, k) for k in OUTPUTS]...)
    ok = vec(all(isfinite, hcat(X, Y); dims=2)); X, Y = X[ok, :], Y[ok, :]
    n = size(X, 1)
    R = [corspearman(X[:, i], Y[:, j]) for i in eachindex(inputs), j in eachindex(OUTPUTS)]
    rng = StableRNG(SEED)
    B = Array{Float64}(undef, N_BOOT, length(inputs), length(OUTPUTS))
    for b in 1:N_BOOT
        idx = rand(rng, 1:n, n)
        for i in eachindex(inputs), j in eachindex(OUTPUTS)
            B[b, i, j] = corspearman(X[idx, i], Y[idx, j])
        end
    end
    S = hcat([py"mean_abs_shap"(X, Y[:, j]) for j in eachindex(OUTPUTS)]...)
    return (n=n, R=R, lo=[quantile(B[:, i, j], 0.025) for i in eachindex(inputs), j in eachindex(OUTPUTS)],
            hi=[quantile(B[:, i, j], 0.975) for i in eachindex(inputs), j in eachindex(OUTPUTS)], S=S)
end

ranks(v) = invperm(sortperm(abs.(v); rev=true))

particles = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]
z_psi = psi_draws(particles)
list = scenarios()
println("SD sweep: $(length(list)) scenarios × $(length(REGULAR_SCHEDULE_CONDITIONS)) conditions × $(length(particles)) particles; " *
        "threads = $(Threads.nthreads())"); flush(stdout)

for (k, (name, varied, scale, dist)) in enumerate(list)
    out = joinpath(SWEEP_DIR, "$(name).csv")
    if isfile(out)
        println("[$k/$(length(list))] $name: already done, skipping"); flush(stdout)
        continue
    end
    t = @elapsed begin
        ps = rescale_particles(particles, scale, dist, z_psi)
        inputs = ALL_INPUTS[analysis_inputs(ps)]   # ψ only in the φ scenarios
        base = build_revision_baseline(ps)
        rows = NamedTuple[]
        excl = Dict{String,Int}()
        for c in REGULAR_SCHEDULE_CONDITIONS
            M = base.by_condition[c].metrics
            excl[c] = count(m -> !m.entrained, M)
            a = analyse_condition(base.by_condition[c].analysis, inputs)
            for (i, inp) in enumerate(inputs), (j, o) in enumerate(OUTPUTS)
                push!(rows, (scenario=name, varied=varied, distribution=String(dist),
                             scale_tau=scale[:tau_x], scale_K=scale[:K], scale_chi=scale[:chi], scale_mu_h=scale[:mu_h],
                             phi_sd_h=scale[:phi], condition=c,
                             n_included=length(base.included_ids), n_fail_this_condition=excl[c],
                             n_not_asleep_every_night=count(m -> !(m.min_night_sleep_fraction > NIGHT_SLEEP_MIN_FRACTION), M),
                             input=String(inp), output=String(o), rho=a.R[i, j], rho_lo=a.lo[i, j], rho_hi=a.hi[i, j],
                             rank_rho=ranks(a.R[:, j])[i], mean_abs_shap=a.S[i, j], rank_shap=ranks(a.S[:, j])[i]))
            end
        end
        CSV.write(out, DataFrame(rows))
    end
    println("[$k/$(length(list))] $name: included $(length(base.included_ids)) " *
            "(fail 150 lux $(excl[REGULAR_SCHEDULE_CONDITION_LOWLIGHT]), 10k $(excl[REGULAR_SCHEDULE_CONDITION_HIGHLIGHT]))  $(round(t, digits=0)) s")
    flush(stdout)
end

all_rows = vcat([CSV.read(joinpath(SWEEP_DIR, "$(s[1]).csv"), DataFrame) for s in list]...)
CSV.write(joinpath(RESULTS_DIR, "sd_sweep_all.csv"), all_rows)
println("\nWrote results/sd_sweep_all.csv ($(nrow(all_rows)) rows)")

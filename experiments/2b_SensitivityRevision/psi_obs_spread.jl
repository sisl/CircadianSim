# Reviewer comment 1.3: how much does sampling ψ widen the observed phase offset ψ_obs (sleep midpoint − CBTmin)?
#
# Re-runs the revision baseline for ψ fixed at 0 and for the ψ scenarios of sd_sweep.jl (ψ = SD · z_ψ, same particles,
# common random numbers, same entrainment exclusions and features), and reports per scenario and condition the spread
# of ψ_obs, CBTmin and sleep timing, the correlation of ψ with ψ_obs, and the number of participants excluded.
# These are the numbers in the response to comment 1.3 (ψ_obs SD 0.63 / 0.59 h at ψ = 0, 1.07 h at a ψ SD of 1 h;
# CBTmin SD 0.43 → 0.53 h at 10,000 lux; 26 → 112 excluded at 150 lux).
# Output: results/psi_obs_spread.csv
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/psi_obs_spread.jl   (~5 min on 8 threads)

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using StatsBase
using DataFrames
using CSV
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const PSI_SD_H = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)   # ψ fixed at 0, then sd_sweep.jl's PHI_SD_H

"""Particles with ψ = sd · z_ψ and every other field unchanged (sd_sweep.jl's ψ scenarios)."""
function with_psi(particles, sd, z_psi)
    map(enumerate(particles)) do (i, p)
        vals = Dict(f => getfield(p, f) for f in fieldnames(CircadianState))
        sd > 0 && (vals[:phi] = sd * z_psi[i])
        CircadianState((vals[f] for f in fieldnames(CircadianState))...)
    end
end

finite_sd(v) = std(filter(isfinite, collect(v)))

particles = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]
z_psi = psi_draws(particles)
println("ψ_obs spread: $(length(PSI_SD_H)) ψ SDs × $(length(REGULAR_SCHEDULE_CONDITIONS)) conditions × $(length(particles)) particles; " *
        "threads = $(Threads.nthreads())"); flush(stdout)

rows = NamedTuple[]
for sd in PSI_SD_H
    t = @elapsed begin
        base = build_revision_baseline(with_psi(particles, sd, z_psi))
        for c in REGULAR_SCHEDULE_CONDITIONS
            f = extract_summary_features(base.by_condition[c].analysis; normalize_time_outputs=true)
            y = collect(f.outputs.psi_obs); ok = isfinite.(y)
            q = quantile(y[ok], [0.05, 0.25, 0.75, 0.95])
            push!(rows, (scenario=sd == 0 ? "baseline" : "phi_sd$(sd)h", psi_sd_h=sd, condition=c,
                         n_included=length(base.included_ids),
                         n_fail_this_condition=count(m -> !m.entrained, base.by_condition[c].metrics),
                         sd_psi_obs_h=std(y[ok]), iqr_psi_obs_h=q[3] - q[2], p5_psi_obs_h=q[1], p95_psi_obs_h=q[4],
                         mean_psi_obs_h=mean(y[ok]), sd_s_mp_h=finite_sd(f.outputs.s_mp), sd_cbtmin_h=finite_sd(f.outputs.cbtmin),
                         sd_s_on_h=finite_sd(f.outputs.s_on), sd_s_off_h=finite_sd(f.outputs.s_off),
                         rho_psi_psi_obs=sd > 0 ? corspearman(collect(f.inputs.psi)[ok], y[ok]) : NaN))
        end
    end
    println("ψ SD $(sd) h: included $(length(base.included_ids))  $(round(t, digits=0)) s"); flush(stdout)
end

df = DataFrame(rows)
CSV.write(joinpath(RESULTS_DIR, "psi_obs_spread.csv"), df)

println("\nψ SD (h)   ψ_obs SD 150 lux / 10,000 lux (h)   CBTmin SD 10,000 lux (h)   excluded at 150 lux")
for sd in PSI_SD_H
    r(c) = only(filter(x -> x.psi_sd_h == sd && x.condition == c, rows))
    lo, hi = r(REGULAR_SCHEDULE_CONDITION_LOWLIGHT), r(REGULAR_SCHEDULE_CONDITION_HIGHLIGHT)
    @printf("  %5.3f    %5.2f / %5.2f                         %5.2f                      %d\n",
            sd, lo.sd_psi_obs_h, hi.sd_psi_obs_h, hi.sd_cbtmin_h, lo.n_fail_this_condition)
end
println("\nWrote results/psi_obs_spread.csv")

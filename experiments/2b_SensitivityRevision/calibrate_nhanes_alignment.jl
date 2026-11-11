# NHANES wake-time alignment for the revision baseline.
# The light schedule's phase (`REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H` in
# src/simulation/light_intensity_core.jl) is calibrated so that the mean simulated sleep offset at
# 150 lux equals the mean NHANES 2021–2023 weekday wake time (SLQ310). This script runs the revision
# baseline (70 d, stochastic light, shared exclusions over both conditions) with the shift
# currently compiled into CircadianPOMDP, reports the mismatch, and suggests the next shift:
#   shift_next = shift_current + (model_mean − NHANES_mean)
# (an entrained rhythm follows the light schedule hour for hour). Edit the constant and rerun until
# the mismatch is below one minute.
# Run: julia -t auto --project=. experiments/2b_SensitivityRevision/calibrate_nhanes_alignment.jl

include(joinpath(@__DIR__, "sensitivity_common.jl"))

using JLD2
using CSV
using DataFrames

"""Clock time "b'HH:MM'" (NHANES CSV export) → hours; NaN if unreadable (as in compareWithNHANES.jl)."""
function nhanes_hours(x)
    s = strip(string(x))
    startswith(s, "b'") && (s = s[3:end-1])
    occursin(":", s) || return NaN
    p = split(s, ":")
    h, m = tryparse(Float64, p[1]), tryparse(Float64, p[2])
    (h === nothing || m === nothing) ? NaN : h + m / 60
end
wrap12(x) = x >= 12 ? x - 24 : x

nh = CSV.read(joinpath(REPO_DATA_ROOT, "NHANES", "SLQ_L_2021_2023.csv"), DataFrame)
wake = filter(isfinite, nhanes_hours.(nh.SLQ310))
nhanes_mean = mean(wrap12.(wake))

shift = CircadianPOMDP.REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H
particles = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]
base = build_revision_baseline(particles)
println("Alignment shift compiled in: $(shift) h (lights on $(round(CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H, digits=3)) h)")
println("NHANES weekday wake time: n = $(length(wake)), mean = $(round(nhanes_mean, digits=4)) h")
println("Included particles: $(length(base.included_ids))")
for c in REGULAR_SCHEDULE_CONDITIONS
    o = wrap12.(mod.([s.sleep_offset_mle for s in base.by_condition[c].analysis], 24))
    d = mean(o) - nhanes_mean
    println("  $c: model mean sleep offset $(round(mean(o), digits=4)) h; model − NHANES = $(round(60d, digits=2)) min")
    c == REGULAR_SCHEDULE_CONDITION_LOWLIGHT &&
        println("  → suggested shift (150 lux calibration): $(round(shift + d, digits=3)) h")
end

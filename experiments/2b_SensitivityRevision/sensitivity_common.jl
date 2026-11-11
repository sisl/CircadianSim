# Shared setup for the sensitivity revision (reviewer comments 2.4, 2.6, 3.6).
# The revision baseline itself (regular schedule with stochastic light, BASELINE_PST_DAYS = 70 scored on days
# MLE_START_DAY–70, entrainment rule, shared particle set) is defined in experiments/revision_baseline.jl
# and loaded through jetlag_reference.jl, so experiments 1–4 and this folder use one definition.
# This file adds what the sensitivity analyses need on top: re-simulating arbitrary particle sets
# (SD sweep, calibration) and experiment 2's 35-day settings, for comparison.

using Dates
using Random
using Statistics
using CircadianPOMDP

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

const REVISION_TOTAL_DAYS = BASELINE_PST_DAYS
const ALL_INPUTS = [:tau, :k, :psi, :chi, :mu_h_over_chi]
const ALL_INPUT_LABELS = ["τ", "k", "ψ", "χ", "μ_H/χ"]

"""
    analysis_inputs(particles) -> BitVector over ALL_INPUTS

The model inputs that vary across `particles` (ψ drops out when fixed at 0; see `varying_inputs`).
"""
analysis_inputs(particles) = varying_inputs([[p.tau_x for p in particles], [p.K for p in particles], [p.phi for p in particles],
                                             [p.chi for p in particles], [p.mu_h / p.chi for p in particles]])
const REVISION_SCORE_START_DAY = MLE_START_DAY

"""
Standard-normal ψ draws of the initial particles: initializeParticles.jl's stream regenerated with phi_sd = 1
(Normal(0, 0) consumes the same draw, so every other field must match the stored particles).
"""
function psi_draws(particles)
    Random.seed!(2025_01_01)   # as initializeParticles.jl
    ref = CircadianPOMDP.initialize_particles(length(particles), START_DATE; phase_mode=:random,
                                              state_params_mode=INITIALIZATION_STATE_PARAMS_MODE, phi_sd=1.0)
    all(r.tau_x == p.tau_x && r.K == p.K && r.chi == p.chi && r.mu_h == p.mu_h && r.x == p.x && r.H == p.H
        for (r, p) in zip(ref, particles)) || error("psi_draws: regenerated particles differ from the initial particles")
    return [r.phi for r in ref]
end

"""
    simulate_regular_schedule(particles, condition; total_days, score_start_day, params, seed)

Simulate (stochastic light, per-particle streams from `seed`) and return the MLE summaries over the scoring window.
Defaults are the revision baseline (70 days, days 56–70).
"""
function simulate_regular_schedule(particles, condition; total_days=REVISION_TOTAL_DAYS,
                                   score_start_day=REVISION_SCORE_START_DAY, params=model_parameters(), seed=42)
    events = simulate_regular_schedule_events(particles, condition; total_days, params, seed)
    return summarize_window(events, particles; score_start_day)
end

"""
    build_revision_baseline(particles; conditions=REGULAR_SCHEDULE_CONDITIONS, seed=42)

Re-simulate the revision baseline for an arbitrary particle set (e.g. an SD-sweep scenario) and apply
the same rule as selectEntrainedParticles.jl: a particle is analysed only if it is entrained and has a
summary in **every** condition. Returns `(included_ids, by_condition)`, where
`by_condition[c] = (summaries, analysis, metrics)`: all summaries, the summaries of included particles
(sorted by particle_id), and the per-particle [`entrainment_metrics`](@ref).
"""
function build_revision_baseline(particles; conditions=REGULAR_SCHEDULE_CONDITIONS, seed=42)
    raw = Dict(c => begin
        events = simulate_regular_schedule_events(particles, c; seed)
        (summaries=summarize_window(events, particles), metrics=entrainment_metrics(events, length(particles)))
    end for c in conditions)
    has_summary(c) = Set(s.particle_id for s in raw[c].summaries)
    included = Set(pid for pid in eachindex(particles)
                   if all(raw[c].metrics[pid].entrained && pid in has_summary(c) for c in conditions))
    by_condition = Dict(c => (summaries=raw[c].summaries,
                              analysis=sort(filter(s -> s.particle_id in included, raw[c].summaries); by=s -> s.particle_id),
                              metrics=raw[c].metrics) for c in conditions)
    return (included_ids=sort(collect(included)), by_condition=by_condition)
end

# --- Experiment 2's 35-day settings ----------------------------------------------------------------
# Used by build_deterministic_baseline.jl to show what the 70-day burn-in changes relative to the published window.
const LEGACY_DRIFT_THRESHOLD_MIN_PER_DAY = 5.0

"""35-day experiment-2 window (days 20.333–35)."""
const LEGACY_WINDOW = (total_days=35, score_start_day=20.333)

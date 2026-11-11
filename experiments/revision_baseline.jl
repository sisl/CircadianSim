# ================================================================
# Revision baseline: regular schedule (stochastic light), entrainment rule, shared particle set
# ================================================================
# Included at the end of jetlag_reference.jl, so every experiment sees the same definitions.
#
# Light: stochastic everywhere (Appendix A.4.1 noise: TruncNormal(I, 0.1 I, 0, 10⁴) lux, drawn each step
# while light is on). Every simulation uses SIMULATION_SEED; each particle gets its own stream
# StableRNG(SIMULATION_SEED + particle_id), so results are reproducible and thread-independent.
#
# Regular schedule: BASELINE_PST_DAYS (70 d) of the regular light schedule, scored on the last 14 days
# (MLE_START_DAY = 56 onward). With the same seed, the on-the-fly regular schedule and the
# precomputed-light jet-lag baseline draw the same noise, so exclusions computed here apply unchanged
# to experiments 3 and 4.
#
# Entrainment rule (evaluated per light condition on the scoring window):
#   (a) asleep for part of every night's dark period (lights off REGULAR_SCHEDULE_LIGHT_OFF_H to
#       lights on REGULAR_SCHEDULE_LIGHT_ON_H): excludes particles that skip nights or sleep only
#       during the day; early and late sleepers pass;
#   (b) CBTmin and sleep-midpoint clock times within ±ENTRAINMENT_BAND_H of their circular means.
# A particle that fails in either condition, or has no MLE summary, is excluded from every analysis.
# 2_RegularScheduleAnalysis/selectEntrainedParticles.jl writes the shared set to
# included_particles_path(); load it with included_particle_ids().

using Statistics
using JLD2

"""Regular-schedule transition (on-the-fly light, stochastic illuminance)."""
regular_schedule_transition(state, action, params; kwargs...) =
    circadian_transition_ext(state, action, params; kwargs..., stochastic_light=true)

"""Jet-lag / intervention transition (precomputed light, stochastic illuminance)."""
jetlag_transition(state, action, params; kwargs...) =
    CircadianPOMDP.transition(state, action, params; kwargs..., deterministic_light=false)   # = circadian_transition_sleep + light noise

"""
    simulate_regular_schedule_events(particles, condition; total_days=BASELINE_PST_DAYS, params, seed)

Simulate `total_days` of the regular schedule for `condition` (stochastic light, SIMULATION_SEED).
Returns `(cbtmin, onsets, offsets)` event vectors.
"""
function simulate_regular_schedule_events(particles, condition; total_days=BASELINE_PST_DAYS,
                                          params=model_parameters(), seed=SIMULATION_SEED, verbose=false)
    light_diet = regular_schedule_light_diet_for_condition(condition)
    action = CircadianActionExt(
        light_diet, REGULAR_SCHEDULE_SLEEP_DIET, REGULAR_SCHEDULE_SOLAR_MODEL,
        DT_HOURS, START_DATE, START_DATE + Day(total_days)
    )
    actions = fill(action, get_action_num_steps(action))
    return simulate_particle_population_events_only(
        particles, actions, params, regular_schedule_transition, START_DATE;
        dt=DT_HOURS, verbose=verbose, seed=seed
    )
end

"""MLE summaries (experiment-2 method) over the window starting at `score_start_day`."""
summarize_window(events, particles; score_start_day=MLE_START_DAY) =
    compute_mle_population_summaries(
        events..., particles;
        dt=DT_HOURS, start_day=score_start_day, min_sleep_episodes=MLE_MIN_SLEEP_EPISODES,
        time_shift=MLE_TIME_SHIFT, verbose=false
    )

# --- Entrainment rule ------------------------------------------------------------------------------
const ENTRAINMENT_BAND_H = 1.0
const NIGHT_SLEEP_MIN_FRACTION = 0.0   # asleep for more than this fraction of every dark period

_circdiff_h(a, b) = mod(a - b + 12, 24) - 12

function circ_mean_h(t)
    θ = 2π .* t ./ 24
    return mod(atan(mean(sin.(θ)), mean(cos.(θ))) * 24 / 2π, 24)
end

"""Largest circular deviation (h) of clock times from their circular mean; NaN if empty."""
max_dev_h(t) = isempty(t) ? NaN : maximum(abs.(_circdiff_h.(t, circ_mean_h(t))))

"""
    night_sleep_fractions(pairs, t0, t1)

Fraction of each night's dark period (lights off `REGULAR_SCHEDULE_LIGHT_OFF_H` to lights on
`REGULAR_SCHEDULE_LIGHT_ON_H` the next morning) spent asleep, for every night lying entirely in days
`[t0, t1)`. `pairs` are onset/offset event pairs from `_pair_onset_offset_events`.
"""
function night_sleep_fractions(pairs, t0, t1)
    off_h, on_h = CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_OFF_H, CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H
    dark_h = mod(on_h - off_h, 24)
    starts = [24d + off_h for d in floor(Int, t0) - 1:ceil(Int, t1)]
    nights = [(s, s + dark_h) for s in starts if s >= 24t0 && s + dark_h <= 24t1]
    return map(nights) do (a, b)
        asleep = sum((max(0.0, min(b, p.offset.t_hours) - max(a, p.onset.t_hours)) for p in pairs); init=0.0)
        asleep / dark_h
    end
end

"""Signed net CBTmin drift (min/day; + = later each day) from CBTmin event times in hours."""
net_drift_min_per_day(t) = length(t) < 3 ? NaN : 60 * mean(diff(t) .- 24)

"""
    entrainment_metrics(events, n_particles; t0=MLE_START_DAY, t1=BASELINE_PST_DAYS)

Per-particle entrainment metrics over days `[t0, t1)`: fraction of each night's dark period asleep
(minimum and mean over the complete nights in the window), max CBTmin and sleep-midpoint deviation (h)
from their circular means, signed net CBTmin drift (min/day), and the `entrained` flag. Also
`days_without_sleep` (calendar days with no sleep onset), kept as a diagnostic only. Returns a vector
indexed by particle id.
"""
function entrainment_metrics(events, n_particles; t0=MLE_START_DAY, t1=Float64(BASELINE_PST_DAYS),
                             band_h=ENTRAINMENT_BAND_H)
    cbtmin, onsets, offsets = events
    group(ev) = (d = Dict{Int,Vector{Any}}(); for e in ev; push!(get!(d, e.particle_id, Any[]), e); end; d)
    cb_by, on_by, off_by = group(cbtmin), group(onsets), group(offsets)
    lo, hi = 24t0, 24t1
    n_bins = floor(Int, t1 - t0)
    return map(1:n_particles) do pid
        cb = sort([e.t_hours for e in get(cb_by, pid, Any[]) if lo <= e.t_hours < hi])
        on = sort(get(on_by, pid, Any[]); by=e -> e.t_hours)
        off = sort(get(off_by, pid, Any[]); by=e -> e.t_hours)
        on_t = [e.t_hours for e in on if lo <= e.t_hours < hi]
        days_without_sleep = n_bins - count(k -> any(x -> lo + 24k <= x < lo + 24(k + 1), on_t), 0:n_bins-1)
        pairs = CircadianPOMDP._pair_onset_offset_events(on, off; start_t_hours=lo)
        mids = [mod((p.onset.t_hours + p.offset.t_hours) / 2, 24) for p in pairs if p.offset.t_hours < hi]
        cb_dev, mid_dev = max_dev_h(mod.(cb, 24)), max_dev_h(mids)
        in_band(d) = isfinite(d) && d <= band_h
        nf = night_sleep_fractions(pairs, t0, t1)
        (particle_id=pid, n_days=n_bins, days_without_sleep=days_without_sleep,
         n_nights=length(nf), min_night_sleep_fraction=isempty(nf) ? NaN : minimum(nf),
         mean_night_sleep_fraction=isempty(nf) ? NaN : mean(nf),
         cbtmin_max_dev_h=cb_dev, midpoint_max_dev_h=mid_dev, drift_min_per_day=net_drift_min_per_day(cb),
         entrained=!isempty(nf) && minimum(nf) > NIGHT_SLEEP_MIN_FRACTION && in_band(cb_dev) && in_band(mid_dev))
    end
end

"""Reason a particle fails the rule in one condition ("" if it passes)."""
function exclusion_reason(m, has_summary::Bool)
    !(m.min_night_sleep_fraction > NIGHT_SLEEP_MIN_FRACTION) && return "not asleep every night"
    !(isfinite(m.cbtmin_max_dev_h) && m.cbtmin_max_dev_h <= ENTRAINMENT_BAND_H) && return "CBTmin outside ±$(ENTRAINMENT_BAND_H) h band"
    !(isfinite(m.midpoint_max_dev_h) && m.midpoint_max_dev_h <= ENTRAINMENT_BAND_H) && return "sleep midpoint outside ±$(ENTRAINMENT_BAND_H) h band"
    !has_summary && return "no MLE summary"
    return ""
end

# --- Shared particle set ---------------------------------------------------------------------------
"""Path to the shared included-particle set (selectEntrainedParticles.jl output)."""
included_particles_path() = joinpath(JETLAG_DATA_ROOT, DIR_REGULAR_SCHEDULE,
                                     "includedParticles_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).jld2")

"""Sorted ids of particles included in every analysis."""
function included_particle_ids()
    p = included_particles_path()
    isfile(p) || error("Included-particle set not found: $p\nRun 2_RegularScheduleAnalysis/selectEntrainedParticles.jl first.")
    return JLD2.load(p, "included_ids")
end

"""Keep only summaries (or anything with `particle_id`) of included particles."""
filter_included(items, ids=Set(included_particle_ids())) = filter(x -> x.particle_id in ids, items)

# --- Settled pre-shift phase (jet-lag adaptation target) ------------------------------------------------
# Some particles are still settling at day 70 (≈14 % at 150 lux: shorter τ, high μ_H/χ; they settle by day ~9 at
# 10,000 lux), so their day-56–70 CBTmin is not yet their entrained phase and "pre-shift CBTmin + shift" misses where
# they re-entrain. The adaptation target uses the CBTmin after SETTLED_REFERENCE_DAYS of the regular light diet
# instead; the simulations themselves still start the shift after BASELINE_PST_DAYS.
const SETTLED_REFERENCE_DAYS = 200

settled_preshift_cbtmin_path(condition_name) = joinpath(regular_schedule_dir(condition=condition_name),
    "settledPreShiftCBTmin_$(SETTLED_REFERENCE_DAYS)days_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_$(CBTMIN_DEFINITION_TAG).jld2")

"""
    settled_preshift_cbtmin(condition_name; compute=true) -> Vector{Union{Float64, Nothing}}

Steady-state CBTmin of every particle (indexed by particle id) after SETTLED_REFERENCE_DAYS of the regular light diet:
precomputed light with jetlag_transition and SIMULATION_SEED, so the first BASELINE_PST_DAYS are the experiment-4
baseline. `nothing` where no steady state is found. Cached at settled_preshift_cbtmin_path; with `compute=false` a
missing cache is an error (concurrent jobs must not all compute it).
"""
function settled_preshift_cbtmin(condition_name; compute::Bool=true)
    path = settled_preshift_cbtmin_path(condition_name)
    isfile(path) && return JLD2.load(path, "settled_cbtmin")
    compute || error("Settled pre-shift CBTmin not found: $path\nRun the experiment-4 baseline stage first.")
    condition = only(filter(c -> c.name == condition_name, light_conditions_specs()))
    particles = JLD2.load(initial_particles_path(N_PARTICLES_DEFAULT), "initial_particles")
    light_diet, _ = generate_light_diets_for_condition(condition; baseline_days=SETTLED_REFERENCE_DAYS, target_days=1)
    events, _, _ = CircadianPOMDP.simulate_particle_population_events_only(particles, light_diet,
        CircadianPOMDP.model_parameters(), jetlag_transition, START_DATE; dt=DT_HOURS, verbose=false, seed=SIMULATION_SEED)
    trajectories, counts = CircadianPOMDP.baseline_cbtmin_from_events(events, length(particles), SETTLED_REFERENCE_DAYS * 24.0)
    settled_cbtmin = Union{Float64, Nothing}[counts[i] >= MIN_CONSECUTIVE_DAYS ?
        CircadianPOMDP.find_baseline_steady_state_cbtmin(trajectories[i]; min_consecutive_days=MIN_CONSECUTIVE_DAYS,
                                                         threshold=Float64(STEADY_STATE_THRESHOLD)) : nothing
        for i in eachindex(particles)]
    mkpath(dirname(path))
    JLD2.jldsave(path * ".partial"; settled_cbtmin, days=SETTLED_REFERENCE_DAYS, seed=SIMULATION_SEED)
    mv(path * ".partial", path; force=true)
    return settled_cbtmin
end

# --- Plotting helper ---------------------------------------------------------------------------------
"""
    unwrap24_centered(x)

`unwrap24` for plotting: unwraps a clock-time trajectory into a continuous curve whose first value lies in
[-12, 12) h (the convention of every other figure), so e.g. a baseline CBTmin of 23:00 plots at -1 h next
to the 05:00 cluster instead of at +23 h. Steady-state detection keeps using `unwrap24`.
"""
function unwrap24_centered(x)
    u = unwrap24(x)
    return isempty(u) ? u : u .- 24 * fld(u[1] + 12, 24)
end

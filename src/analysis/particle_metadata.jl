# ================================================================
# Particle Event Utilities
# ================================================================
# Functions for processing CBTmin and sleep events produced by
# simulate_particle_population_events_only.

using Statistics

# ================================================================
# SLEEP AND CIRCADIAN METRICS STRUCTURES (for events-based pipeline)
# ================================================================

"""
    SleepMetrics

Struct to hold comprehensive sleep-related metrics for a particle (from events).
"""
struct SleepMetrics
    sleep_periods::Vector{SleepPeriod}
    sleep_onsets::Vector{Int}
    sleep_offsets::Vector{Int}
    sleep_midpoints::Vector{Int}
    sleep_durations::Vector{Float64}
    total_sleep_time::Float64
    mean_sleep_duration::Float64
    sleep_percent::Float64
    num_sleep_episodes::Int
end

"""
    CircadianMetrics

Struct to hold circadian rhythm metrics for a particle (from events).
"""
struct CircadianMetrics
    cbtmin_indices::Vector{Int}
    cbtmin_times::Vector{Float64}
    inter_cbtmin_intervals::Vector{Float64}
    mean_cbtmin_period::Float64
    circadian_amplitude::Float64
    phase_drift::Float64
    num_cbtmin_events::Int
    circadian_regularity::Float64
end

"""
    baseline_cbtmin_from_events(cbtmin_events, num_particles::Int, baseline_hours::Real)

Build baseline CBTmin data from events for downstream JetLag scripts.
Returns `(baseline_cbtmin_trajectories, baseline_cbtmin_counts)` where:
- `baseline_cbtmin_trajectories[i]` = clock_h values for particle i's CBTmin events during baseline
- `baseline_cbtmin_counts[i]` = number of CBTmin events during baseline for particle i
"""
function baseline_cbtmin_from_events(cbtmin_events::AbstractVector, num_particles::Int, baseline_hours::Real)
    trajectories = Vector{Vector{Float64}}(undef, num_particles)
    counts = Vector{Int}(undef, num_particles)
    for pid in 1:num_particles
        particle_events = [e for e in cbtmin_events if e.particle_id == pid && e.t_hours <= baseline_hours]
        sort!(particle_events, by=e -> e.t_hours)
        trajectories[pid] = [e.clock_h for e in particle_events]
        counts[pid] = length(particle_events)
    end
    return trajectories, counts
end

"""
    load_particle_histories(path::AbstractString)

Load particle history JLD2 file (events-based format from `simulate_particle_population_events_only`).
Handles `particles` vs `particles_ext` and optional `num_particles` key.

Returns a named tuple `(cbtmin_events, sleep_onset_events, sleep_offset_events, particles, num_particles)`.
"""
function load_particle_histories(path::AbstractString)
    data = JLD2.load(path)
    cbtmin_events = data["cbtmin_events"]
    sleep_onset_events = get(data, "sleep_onset_events", [])
    sleep_offset_events = get(data, "sleep_offset_events", [])
    particles = get(data, "particles") do
        data["particles_ext"]
    end
    num_particles = get(data, "num_particles", length(particles))
    return (; cbtmin_events, sleep_onset_events, sleep_offset_events, particles, num_particles)
end

"""
    load_particle_summaries(path::AbstractString)

Load particle MLE summaries from a JLD2 file. Returns `data["summaries"]`.
"""
function load_particle_summaries(path::AbstractString)
    data = JLD2.load(path)
    return data["summaries"]
end

"""
    extract_summary_features(summaries; normalize_time_outputs::Bool=false) -> NamedTuple

Extract input parameters and output variables from MLE particle summaries for
cross-correlation, SHAP, and adaptation analysis.

**Inputs (5):** tau, K, phi, chi, mu_h/chi  
**Outputs (6):** sleep_onset_mle, sleep_offset_mle, sleep_duration_mle, sleep_midpoint_mle,
CBTmin_mle, phase_offset

Returns a NamedTuple with:
- `inputs`: `(tau, k, psi, chi, mu_h_over_chi)` as `Vector{Float64}`
- `outputs`: `(s_on, s_off, s_dur, s_mp, cbtmin, psi_obs)` as `Vector{Float64}`
- `input_names`, `output_names`: string vectors for labeling

If `normalize_time_outputs=true`, applies `normalize_time!` to s_on, s_off, s_mp, cbtmin
(sleep duration and phase_offset are not normalized).
"""
function extract_summary_features(summaries; normalize_time_outputs::Bool=false)
    tau_values = Float64[]
    k_values = Float64[]
    psi_values = Float64[]
    chi_values = Float64[]
    mu_h_over_chi_values = Float64[]
    s_on_values = Float64[]
    s_off_values = Float64[]
    s_dur_values = Float64[]
    s_mp_values = Float64[]
    cbtmin_values = Float64[]
    psi_obs_values = Float64[]

    for s in summaries
        try
            push!(tau_values, s.tau)
            push!(k_values, s.K)
            push!(psi_values, s.phi)
            push!(chi_values, s.chi)
            push!(mu_h_over_chi_values, s.mu_h / s.chi)
            push!(s_on_values, s.sleep_onset_mle)
            push!(s_off_values, s.sleep_offset_mle)
            push!(s_dur_values, s.sleep_duration_mle)
            push!(s_mp_values, s.sleep_midpoint_mle)
            push!(cbtmin_values, s.CBTmin_mle)
            push!(psi_obs_values, s.phase_offset)
        catch e
            @warn "Error extracting data from summary: $e"
        end
    end

    if normalize_time_outputs
        normalize_time!(s_on_values)
        normalize_time!(s_off_values)
        normalize_time!(s_mp_values)
        normalize_time!(cbtmin_values)
    end

    input_names = ["tau", "k", "psi", "chi", "mu_h_over_chi"]
    output_names = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "psi_obs"]

    return (;
        inputs = (tau = tau_values, k = k_values, psi = psi_values, chi = chi_values, mu_h_over_chi = mu_h_over_chi_values),
        outputs = (s_on = s_on_values, s_off = s_off_values, s_dur = s_dur_values, s_mp = s_mp_values, cbtmin = cbtmin_values, psi_obs = psi_obs_values),
        input_names, output_names,
    )
end

"""
    circular_mean_clock_h(hours::Vector{Float64}) -> Float64

Mean of clock hours (0–24) treating the scale as circular (wrapped).
Uses angular mean: convert to angle, average sin/cos, convert back to [0, 24).
"""
function circular_mean_clock_h(hours::Vector{Float64})
    isempty(hours) && return NaN
    θ = (hours ./ 24.0) .* (2.0 * π)
    mean_sin = sum(sin, θ) / length(θ)
    mean_cos = sum(cos, θ) / length(θ)
    a = atan(mean_sin, mean_cos)  # (-π, π]
    a_02pi = mod(a + 2.0 * π, 2.0 * π)  # [0, 2π)
    return a_02pi * 12.0 / π  # [0, 24) clock hours
end

"""
    mean_sleep_midpoint_clock_h_from_events(sleep_onset_events, sleep_offset_events, num_particles::Int;
                                            t_start_hours::Real=0.0, t_end_hours::Real=Inf) -> Vector{Float64}

From sleep onset/offset events in the time window `[t_start_hours, t_end_hours]`, compute per-particle
mean sleep midpoint in clock hours (0–24). Each sleep period midpoint = `mod(onset_clock_h + duration/2, 24)`.
The mean over periods is computed as a **circular mean** (wrapped) so e.g. 23h and 1h average to ~0h, not 12h.
Returns a vector of length `num_particles`; particles with no sleep in the window get `NaN`.
"""
function mean_sleep_midpoint_clock_h_from_events(sleep_onset_events::AbstractVector, sleep_offset_events::AbstractVector,
                                                 num_particles::Int;
                                                 t_start_hours::Real=0.0, t_end_hours::Real=Inf)
    out = fill(NaN, num_particles)
    t_start = Float64(t_start_hours)
    t_end = Float64(t_end_hours)
    for pid in 1:num_particles
        onsets = [e for e in sleep_onset_events if e.particle_id == pid && e.t_hours >= t_start && e.t_hours < t_end]
        offsets = [e for e in sleep_offset_events if e.particle_id == pid && e.t_hours >= t_start && e.t_hours < t_end]
        sort!(onsets, by=e -> e.t_hours)
        sort!(offsets, by=e -> e.t_hours)
        pairs = _pair_onset_offset_events(onsets, offsets)
        midpoints = Float64[mod(p.onset.clock_h + (p.offset.t_hours - p.onset.t_hours) / 2.0, 24.0) for p in pairs]
        if !isempty(midpoints)
            out[pid] = circular_mean_clock_h(midpoints)
        end
    end
    return out
end

"""
    _pair_onset_offset_events(onsets, offsets; start_t_hours=nothing)

Pair sorted sleep-onset and sleep-offset event lists into matched sleep episodes.
Each onset is paired with the next offset that occurs after it; offsets with no
preceding onset are skipped.

If `start_t_hours` is provided, onsets whose `t_hours < start_t_hours` are excluded
before pairing. Offsets that precede any included onset are also skipped.

Events must have a `.t_hours` field. Returns a `Vector` of `(onset=..., offset=...)`
named tuples.
"""
function _pair_onset_offset_events(onsets, offsets; start_t_hours=nothing)
    filtered_onsets = start_t_hours !== nothing ? filter(e -> e.t_hours >= start_t_hours, onsets) : onsets
    pairs = []
    i, j = 1, 1
    while i <= length(filtered_onsets) && j <= length(offsets)
        if filtered_onsets[i].t_hours < offsets[j].t_hours
            push!(pairs, (onset=filtered_onsets[i], offset=offsets[j]))
            i += 1
            j += 1
        else
            j += 1  # this offset precedes the current onset; skip it
        end
    end
    return pairs
end

"""
    events_to_trajectories(events, num_particles::Int)

Build per-particle CBTmin trajectories (ordered clock_h values) from a flat event list.
Events must have `.particle_id`, `.t_hours`, `.clock_h`. Returns `Vector{Vector{Float64}}` indexed by particle id.
"""
function events_to_trajectories(events::AbstractVector, num_particles::Int)
    trajectories = Vector{Vector{Float64}}(undef, num_particles)
    for pid in 1:num_particles
        particle_events = [e for e in events if e.particle_id == pid]
        sort!(particle_events, by=e -> e.t_hours)
        trajectories[pid] = [e.clock_h for e in particle_events]
    end
    return trajectories
end

"""
    cbtmin_events_to_trajectory_list(cbtmin_events, num_particles::Int)

Convert `cbtmin_events` to a list of named tuples expected by adaptation-time logic.
Each element has fields `(particle_id, cbtmin_circadian, cbtmin_absolute)`. Only particles with
at least one CBTmin event are included.
"""
function cbtmin_events_to_trajectory_list(cbtmin_events::AbstractVector, num_particles::Int)
    trajs = Vector{Vector{Float64}}(undef, num_particles)
    trajs_abs = Vector{Vector{Float64}}(undef, num_particles)
    for pid in 1:num_particles
        particle_events = [e for e in cbtmin_events if e.particle_id == pid]
        sort!(particle_events, by=e -> e.t_hours)
        trajs[pid] = [e.clock_h for e in particle_events]
        trajs_abs[pid] = [e.t_hours for e in particle_events]
    end
    out = []
    for i in 1:num_particles
        if !isempty(trajs[i])
            push!(out, (
                particle_id = i,
                cbtmin_circadian = trajs[i],
                cbtmin_absolute = trajs_abs[i]
            ))
        end
    end
    return out
end

"""
    cbtmin_events_from_state_histories(particle_histories, dt; start_date)

Extract `cbtmin_events` from full state histories in the same format as
`simulate_particle_population_events_only`: `(particle_id, step_idx, clock_h, t_hours, datetime)`.
"""
function cbtmin_events_from_state_histories(particle_histories::AbstractVector, dt::Real;
                                            start_date::Dates.DateTime=Dates.DateTime(2023, 1, 1, 0, 0, 0))
    events = []
    for (pid, states) in enumerate(particle_histories)
        idxs = get_CBTmin(states; dt=Float64(dt))
        for idx in idxs
            t_hours = (idx - 1) * Float64(dt)
            clock_h = mod(t_hours, 24.0)
            datetime_val = start_date + Dates.Millisecond(round(Int, t_hours * 3600 * 1000))
            push!(events, (
                particle_id=pid,
                step_idx=idx,
                clock_h=clock_h,
                t_hours=t_hours,
                datetime=datetime_val
            ))
        end
    end
    sort!(events, by=e -> (e.particle_id, e.t_hours))
    return events
end

"""
    merge_baseline_and_post_events(baseline_cbtmin_events, post_events, baseline_hours)

Merge baseline CBTmin events with post-baseline (e.g. intervention or target) events.
Post events have `t_hours` in `[0, ...]`; they are offset by `baseline_hours` so the
combined trajectory is continuous in time. Returns a single sorted event list.
"""
function merge_baseline_and_post_events(baseline_cbtmin_events::AbstractVector, post_events::AbstractVector, baseline_hours::Real)
    bh = Float64(baseline_hours)
    offset_events = [
        (particle_id=e.particle_id, step_idx=e.step_idx, clock_h=e.clock_h,
         t_hours=e.t_hours + bh, datetime=e.datetime)
        for e in post_events
    ]
    combined = vcat(baseline_cbtmin_events, offset_events)
    sort!(combined, by=e -> (e.particle_id, e.t_hours))
    return combined
end

"""
    circadian_metrics_from_events(cbtmin_events, particle_id::Int; simulation_duration=nothing)

Build CircadianMetrics for one particle from CBTmin events (from on-the-fly simulation).
"""
function circadian_metrics_from_events(cbtmin_events, particle_id::Int; simulation_duration=nothing)
    particle_events = [e for e in cbtmin_events if e.particle_id == particle_id]
    sort!(particle_events, by=e -> e.t_hours)
    cbtmin_indices = [e.step_idx for e in particle_events]
    cbtmin_times = [e.t_hours for e in particle_events]
    inter_cbtmin_intervals = length(cbtmin_times) > 1 ? diff(cbtmin_times) : Float64[]
    mean_cbtmin_period = isempty(inter_cbtmin_intervals) ? 24.0 : mean(inter_cbtmin_intervals)
    circadian_regularity = length(inter_cbtmin_intervals) > 1 ? max(0.0, 1.0 - std(inter_cbtmin_intervals) / mean(inter_cbtmin_intervals)) : 0.0
    return CircadianMetrics(
        cbtmin_indices, cbtmin_times, inter_cbtmin_intervals, mean_cbtmin_period,
        0.0, 0.0, length(cbtmin_indices), circadian_regularity
    )
end

"""
    sleep_metrics_from_events(sleep_onset_events, sleep_offset_events, particle_id::Int;
                              simulation_duration=nothing)

Build SleepMetrics for one particle from sleep onset/offset events.
"""
function sleep_metrics_from_events(sleep_onset_events, sleep_offset_events, particle_id::Int;
                                   simulation_duration=nothing)
    onsets = [e for e in sleep_onset_events if e.particle_id == particle_id]
    offsets = [e for e in sleep_offset_events if e.particle_id == particle_id]
    sort!(onsets, by=e -> e.t_hours)
    sort!(offsets, by=e -> e.t_hours)
    sleep_periods = SleepPeriod[]
    for p in _pair_onset_offset_events(onsets, offsets)
        onset_idx  = p.onset.step_idx
        offset_idx = p.offset.step_idx
        dur        = p.offset.t_hours - p.onset.t_hours
        mid_idx    = round(Int, onset_idx + (offset_idx - onset_idx) / 2)
        push!(sleep_periods, SleepPeriod(onset_idx, offset_idx, dur, mid_idx))
    end
    sleep_onsets = Int[sp.onset_idx for sp in sleep_periods]
    sleep_offsets = Int[sp.offset_idx for sp in sleep_periods]
    sleep_midpoints = Int[sp.midpoint_idx for sp in sleep_periods]
    sleep_durations = Float64[sp.duration for sp in sleep_periods]
    total_sleep_time = isempty(sleep_durations) ? 0.0 : sum(sleep_durations)
    mean_sleep_duration = isempty(sleep_durations) ? 0.0 : mean(sleep_durations)
    sim_dur = something(simulation_duration, 0.0)
    sleep_percent = sim_dur > 0 ? total_sleep_time / sim_dur : 0.0
    return SleepMetrics(
        sleep_periods, sleep_onsets, sleep_offsets, sleep_midpoints,
        sleep_durations, total_sleep_time, mean_sleep_duration, sleep_percent, length(sleep_periods)
    )
end

"""
    build_metrics_from_events(cbtmin_events, sleep_onset_events, sleep_offset_events,
                              num_particles::Int; simulation_duration=nothing)

Build circadian_metrics_per_particle and sleep_metrics_per_particle vectors from on-the-fly
event lists. Returns (circadian_metrics_per_particle, sleep_metrics_per_particle).
"""
function build_metrics_from_events(cbtmin_events, sleep_onset_events, sleep_offset_events,
                                   num_particles::Int; simulation_duration=nothing)
    circadian_metrics_per_particle = Vector{CircadianMetrics}(undef, num_particles)
    sleep_metrics_per_particle = Vector{SleepMetrics}(undef, num_particles)
    for pid in 1:num_particles
        circadian_metrics_per_particle[pid] = circadian_metrics_from_events(
            cbtmin_events, pid; simulation_duration=simulation_duration)
        sleep_metrics_per_particle[pid] = sleep_metrics_from_events(
            sleep_onset_events, sleep_offset_events, pid; simulation_duration=simulation_duration)
    end
    return circadian_metrics_per_particle, sleep_metrics_per_particle
end

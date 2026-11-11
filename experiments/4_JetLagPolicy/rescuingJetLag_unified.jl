#! /usr/bin/env julia

# ================================================================
# Unified Rescuing Jet Lag (Policy Search) – Steady-State Version
# ================================================================
# Design:
#   - **First timezone (BASELINE_PST_DAYS):** Same as regular schedule analysis
#     (2_RegularScheduleAnalysis): same photoperiod as Regular_V4 (REGULAR_SCHEDULE_LIGHT_ON_H–OFF; NHANES shift REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H).
#   - **Baseline jet lag:** Same as experiments/3_JetLagAnalysis/generateJetLagParticleHistories.jl:
#     BASELINE_PST_DAYS first timezone → post-shift window (INTERVENTION_TARGET_DAYS) in second (target) timezone (see jetlag_reference.jl), no intervention.
#   - **Intervention:** For each jet lag condition we insert a **single day** between
#     the BASELINE_PST_DAYS in the first timezone and the target segment in the second. That day
#     delivers a light intervention: 10,000 lux for 1–24 hours (swept), all other
#     times dark. Start times swept from midnight (0) to 11 pm (23). Each window
#     stays within the same day (start + duration ≤ 24 h; no wraparound). Hours are on the
#     simulation clock, so the intervention day is 00:00–24:00 (see intervention_day_diet).
#   - Adaptation time is the first day after the shift when CBTmin reaches steady
#     state (steady‐state detection from detectSteadyState.jl) at the target = settled pre-shift
#     CBTmin (settled_preshift_cbtmin, revision_baseline.jl) + shift; we compare baseline
#     vs intervention median adaptation times. Post-shift window per condition:
#     INTERVENTION_TARGET_DAYS (150 lux 100 d, 10,000 lux 50 d).
#
# Particle histories: Data/4_JetLagIntervention/particleHistories/
# Results: Data/4_JetLagIntervention/Adaptations/Particles{N}/{Condition}/
#   systematic_rescue_analysis_steadyState.jld2
#
# Mini test: run with argument "mini" or ENV RESCUE_MINI=1 for a short run
# (50 particles, 2 TZ, 3 policies, 20-day target, 1 condition). Outputs go to
# 4_JetLagIntervention/Adaptations/Mini/ and 4_JetLagIntervention/Mini/particleHistories/.
#
# Cluster runs (sherlock/exp4_sherlock.sh) split the work into stages; by default one process runs everything.
#   RESCUE_STAGE=baseline   post-baseline states and no-intervention results only
#   RESCUE_STAGE=policies   only the policies in RESCUE_POLICIES (e.g. "1:10"). Needs the baseline stage's files and
#                           writes per-policy files only, so many jobs can run at once.
#   RESCUE_STAGE=aggregate  merges the per-policy files (all must exist) into systematic_rescue_analysis_steadyState.jld2
#   RESCUE_CONDITION=LowLight150lux (or HighLight10000lux) restricts any stage to one condition.
# Every simulation uses SIMULATION_SEED with per-particle RNGs, so the split doesn't change the results.
#
# Lara Weed
# Steady-state policy version: December 2025

# Mini-mode flag and paths (from jetlag_reference.jl)
println("🚀 Starting Unified Jet Lag Rescue Analysis (Steady-State Adaptation)")
println("📦 Setting up packages...")

using CircadianPOMDP
import CircadianPOMDP: model_parameters,
                       simulate_particle_population_events_only,
                       simulate_particle_from_initial_population,
                       circadian_transition_sleep, baseline_cbtmin_from_events,
                       cbtmin_events_to_trajectory_list,
                       cbtmin_events_from_state_histories, merge_baseline_and_post_events

# Light diet helpers: generate_light_diets_for_condition, get_intervention_policies (CircadianPOMDP/light_diet_functions.jl); intervention_day_diet below

using JLD2
using Plots
using Statistics
using Random
using StatsBase
using Printf
using Dates
using Base.Threads

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
const RESCUE_MINI = (length(ARGS) > 0 && ARGS[1] == "mini") || get(ENV, "RESCUE_MINI", "0") == "1"
# Per-policy post-shift events are saved compactly (~0.5 MB each, ~7 GB for a full run); RESCUE_SAVE_HISTORIES=0
# turns this off. No-intervention baseline events are always saved.
const SAVE_INTERVENTION_HISTORIES = get(ENV, "RESCUE_SAVE_HISTORIES", "1") == "1"
const PARTICLE_HIST_DIR = intervention_particle_hist_dir(mini=RESCUE_MINI)
const RESCUE_STAGE = get(ENV, "RESCUE_STAGE", "all")
RESCUE_STAGE in ("all", "baseline", "policies", "aggregate") ||
    error("RESCUE_STAGE must be all, baseline, policies or aggregate (got \"$(RESCUE_STAGE)\")")
const RESCUE_CONDITION = get(ENV, "RESCUE_CONDITION", "")
println("   Stage: $(RESCUE_STAGE)" * (isempty(RESCUE_CONDITION) ? "" : ", condition $(RESCUE_CONDITION)"))

if RESCUE_MINI
    println("🧪 MINI TEST MODE (fewer particles, TZ, policies, shorter target)")
    println("   Threading: $(Threads.nthreads()) thread(s). Use 'julia -t auto' for parallel mini run.")
end

Random.seed!(2025_04_01)
println("   ✓ Packages loaded")

# Ensure Data/JetLag directory layout exists (same as generateJetLagParticleHistories.jl)
ensure_jetlag_data_dirs()

# ================================================================
# DATA PATHS AND SIMULATION SETTINGS (from jetlag_reference.jl)
# ================================================================
# START_DATE, DT_HOURS, BASELINE_PST_DAYS, BASELINE_PST_HOURS, light_conditions_specs()

light_conditions = light_conditions_specs()

println("🔬 Will test $(length(light_conditions)) light conditions:")
for condition in light_conditions
    println("   • $(condition.description)")
    println("     Baseline: $(condition.baseline_lux) lux, Target: $(condition.target_lux) lux, Intervention: $(condition.intervention_lux) lux")
end
println()

# ================================================================
# INCREMENTAL SAVING HELPERS
# ================================================================

"""Write `path` via `f(tmp_path)` and rename it into place, so a job killed mid-write leaves no truncated file."""
function write_atomically(f, path)
    tmp_path = path * ".partial"
    f(tmp_path)
    mv(tmp_path, path; force=true)
end

"""
Save intermediate results to avoid data loss during long computations.
"""
function save_checkpoint(save_dir, checkpoint_name, data)
    mkpath(save_dir)
    checkpoint_path = joinpath(save_dir, "$(checkpoint_name).jld2")
    try
        write_atomically(p -> jldsave(p; data), checkpoint_path)
        println("   ✓ Checkpoint saved: $(checkpoint_name)")
        return true
    catch e
        @warn "Failed to save checkpoint $(checkpoint_name): $e"
        return false
    end
end

"""
Load checkpoint if it exists.
"""
function load_checkpoint(save_dir, checkpoint_name)
    checkpoint_path = joinpath(save_dir, "$(checkpoint_name).jld2")
    if isfile(checkpoint_path)
        try
            data = load(checkpoint_path)
            println("   ✓ Loaded checkpoint: $(checkpoint_name)")
            return data
        catch e
            @warn "Failed to load checkpoint $(checkpoint_name): $e"
            return nothing
        end
    end
    return nothing
end

"""
Save policy results incrementally.
"""
function save_policy_results_incremental(save_dir, policy_idx, policy_results, policy)
    policy_data = Dict(
        "policy_idx" => policy_idx,
        "policy" => policy,
        "results_by_timezone" => policy_results,
        "timestamp" => string(now())
    )

    checkpoint_name = "policy_$(policy_idx)_results_steadyState"
    return save_checkpoint(save_dir, checkpoint_name, policy_data)
end

"""
Merge `policy_k_results_steadyState.jld2` into `intervention_results` for any index `k` in
`policy_indices` not present yet. Cumulative checkpoints are only written every N policies, so the
latest per-policy files can be ahead of the cumulative file on disk after an interrupt.
"""
function merge_individual_policy_checkpoints!(intervention_results, save_dir, policy_indices)
    loaded = 0
    for k in policy_indices
        haskey(intervention_results, k) && continue
        path = joinpath(save_dir, "policy_$(k)_results_steadyState.jld2")
        isfile(path) || continue
        try
            raw = load(path)
            pd = raw["data"]
            intervention_results[k] = (
                policy = pd["policy"],
                results_by_timezone = pd["results_by_timezone"],
            )
            loaded += 1
        catch e
            @warn "Skipping corrupt policy checkpoint policy_$(k)_results_steadyState: $e"
        end
    end
    loaded > 0 &&
        println("   ✓ Merged $(loaded) per-policy checkpoint(s) from disk (ahead of last cumulative save)")
    return intervention_results
end

"""
Check if we can resume from previous run and load existing results for `policy_indices`, so
per-policy files can fill gaps after the last cumulative snapshot. The cumulative snapshot is
only read when `load_cumulative` (it is written by single-process runs only).
"""
function check_resume_capability(save_dir, policy_indices; load_cumulative::Bool=true)
    baseline_checkpoint = load_checkpoint(save_dir, "baseline_results_steadyState")
    cumulative_checkpoint = load_cumulative ?
        load_checkpoint(save_dir, "intervention_results_cumulative_steadyState") : nothing

    intervention_results = if cumulative_checkpoint !== nothing
        filter(kv -> kv[1] in policy_indices, cumulative_checkpoint["data"])
    else
        Dict{Int,Any}()
    end
    merge_individual_policy_checkpoints!(intervention_results, save_dir, policy_indices)
    has_interventions = !isempty(intervention_results)

    return (
        baseline_results = baseline_checkpoint !== nothing ? baseline_checkpoint["data"] : nothing,
        intervention_results = intervention_results,
        can_resume_baseline = baseline_checkpoint !== nothing,
        can_resume_interventions = has_interventions
    )
end

# ================================================================
# INTERVENTION POLICIES (from CircadianPOMDP/light_diet_functions.jl)
# ================================================================

println("🎯 Defining systematic intervention policies (timing sweep)...")
intervention_policies = get_intervention_policies()
if RESCUE_MINI
    intervention_policies = intervention_policies[1:3]
    println("   [MINI] Using $(length(intervention_policies)) policies")
end
println("   Generated $(length(intervention_policies)) intervention policies (start + duration ≤ 24 h)")
# Policies this process runs (stage "policies"); every other stage covers all of them
policy_indices = if RESCUE_STAGE == "policies" && haskey(ENV, "RESCUE_POLICIES")
    first_k, last_k = parse.(Int, split(ENV["RESCUE_POLICIES"], ":"))
    collect(max(first_k, 1):min(last_k, length(intervention_policies)))
else
    collect(eachindex(intervention_policies))
end
RESCUE_STAGE == "policies" && println("   Running policies $(first(policy_indices))–$(last(policy_indices))")
println()

# ================================================================
# STEADY-STATE HELPER FUNCTIONS
# ================================================================
# Core functions (unwrap24, wrap24, is_steady_state_window, find_baseline_steady_state_cbtmin,
# find_first_steady_state_day, compute_adaptation_time) are imported from CircadianPOMDP.
# STEADY_STATE_THRESHOLD, MIN_CONSECUTIVE_DAYS, EXPECTED_CBTMIN_TOLERANCE from jetlag_reference.jl

"""
Compute adaptation time (in days) and adaptation direction from a single
CBTmin trajectory NamedTuple using the package-level steady-state method.
Looks up baseline data from REGULAR_BASELINE_COUNTS / REGULAR_BASELINE_CBTMINS; the target is
TARGET_REFERENCE_CBTMINS (settled pre-shift CBTmin) + shift.

Returns a NamedTuple `(time, direction, baseline_cbtmin, steady_cbtmin)` or `nothing`.
"""
function compute_adaptation_time_steady_state(cbtmin_traj, tz_shift;
                                              baseline_days::Int=BASELINE_PST_DAYS)
    cbtmin_circadian = cbtmin_traj.cbtmin_circadian

    isempty(cbtmin_circadian) && return nothing

    particle_id = cbtmin_traj.particle_id
    if particle_id < 1 || particle_id > length(REGULAR_BASELINE_COUNTS)
        return nothing
    end
    particle_id in INCLUDED_IDS || return nothing   # shared particle set (revision)

    baseline_count = REGULAR_BASELINE_COUNTS[particle_id]
    REGULAR_BASELINE_CBTMINS[particle_id] === nothing && return nothing

    return compute_adaptation_time(
        cbtmin_circadian,
        TARGET_REFERENCE_CBTMINS[particle_id],
        baseline_count,
        tz_shift;
        min_consecutive_days=MIN_CONSECUTIVE_DAYS,
        threshold=Float64(STEADY_STATE_THRESHOLD),
        expected_tolerance=Float64(EXPECTED_CBTMIN_TOLERANCE)
    )
end

# ================================================================
# EVENTS-BASED TRAJECTORY HELPERS (same format as generateJetLagParticleHistories.jl)
# ================================================================

# events_to_trajectories, cbtmin_events_to_trajectory_list, cbtmin_events_from_state_histories,
# merge_baseline_and_post_events: from CircadianPOMDP (src/particle_metadata.jl). We use
# cbtmin_events_to_trajectory_list for adaptation-time trajectory format.

# ================================================================
# POST-BASELINE SEEDING (run the BASELINE_PST_DAYS PST baseline once, reuse for all scenarios)
# ================================================================

"""
Load or compute post-baseline particles and baseline CBTmin events (BASELINE_PST_DAYS PST).
Events-only simulation with `return_final_states=true` (no full state histories), stochastic light and
SIMULATION_SEED, so the baseline is identical to the regular-schedule run used for the shared particle set.
Saves to particleHistories/postBaseline_{BASELINE_PST_DAYS}daysPST_{condition}_{alignment}.jld2.
Returns (post_baseline_particles::Vector, baseline_cbtmin_events).
"""
function load_or_compute_post_baseline(condition, LD_PST, initial_particles, params)
    post_baseline_path = joinpath(PARTICLE_HIST_DIR,
        "postBaseline_$(BASELINE_PST_DAYS)daysPST_$(condition.name)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)$(RESCUE_MINI ? "_mini" : "").jld2")

    if isfile(post_baseline_path)
        data = load(post_baseline_path)
        post_particles = data["post_baseline_particles"]
        baseline_events = data["baseline_cbtmin_events"]
        println("   ✓ Loaded post-baseline particles and events from $(basename(post_baseline_path))")
        return post_particles, baseline_events
    end
    # Concurrent policy jobs must not all compute (and write) the shared baseline
    RESCUE_STAGE in ("policies", "aggregate") &&
        error("Post-baseline file not found: $(post_baseline_path)\nRun RESCUE_STAGE=baseline first.")

    println("   🔄 Running $(BASELINE_PST_DAYS) days PST (events only + final states) to get post-baseline states and events...")
    baseline_cbtmin_events, _, _, post_baseline_particles = simulate_particle_population_events_only(
        initial_particles, LD_PST, params, jetlag_transition, START_DATE;
        dt=DT_HOURS, verbose=false, force_threading=RESCUE_MINI, seed=SIMULATION_SEED, return_final_states=true
    )
    n_particles = length(post_baseline_particles)

    mkpath(dirname(post_baseline_path))
    write_atomically(post_baseline_path) do p
        jldsave(p; post_baseline_particles, baseline_cbtmin_events, num_particles=n_particles)
    end
    println("   ✓ Saved post-baseline particles and baseline events to $(basename(post_baseline_path))")
    return post_baseline_particles, baseline_cbtmin_events
end

# merge_baseline_and_post_events: from CircadianPOMDP (src/particle_metadata.jl)
# generate_light_diets_for_condition, build_baseline_diet_full, build_intervention_diet_full: from CircadianPOMDP (light_diet_functions.jl)

# ================================================================
# LIGHT DIET GENERATION / SIMULATION HELPERS
# ================================================================

"""
Simulate using events-only format (faster, O(events) memory) and extract CBTmin trajectories.
If `save_path` is provided and the file exists, loads from it; otherwise runs
simulate_particle_population_events_only and optionally saves to save_path.
Returns list of (particle_id, cbtmin_circadian, cbtmin_absolute) for particles with at least one CBTmin.
"""
function simulate_and_extract_cbtmin_events(light_diet, initial_particles, params; save_path=nothing)
    n_particles = length(initial_particles)
    transition_fn = jetlag_transition

    if save_path !== nothing && isfile(save_path)
        data = load(save_path)
        cbtmin_events = data["cbtmin_events"]
        n = haskey(data, "num_particles") ? data["num_particles"] : n_particles
        return cbtmin_events_to_trajectory_list(cbtmin_events, n)
    end

    cbtmin_events, _, _ = simulate_particle_population_events_only(
        initial_particles, light_diet, params, transition_fn, START_DATE;
        dt=DT_HOURS, verbose=false, force_threading=RESCUE_MINI, seed=SIMULATION_SEED
    )
    out = cbtmin_events_to_trajectory_list(cbtmin_events, n_particles)

    if save_path !== nothing
        mkpath(dirname(save_path))
        simulation_duration = length(light_diet) * DT_HOURS
        num_particles = n_particles
        start_date = START_DATE
        dt_hours = DT_HOURS
        @save save_path cbtmin_events num_particles simulation_duration start_date dt_hours
    end

    return out
end

"""
Light for the 24-h intervention day (the first day after the shift): `lux` with forced wake from `policy.start` to
`policy.start + policy.duration` h, 0 lux with open sleep/wake otherwise. Hours are on the simulation clock, so every
window is one contiguous block inside the day (start 0–23 h, start + duration ≤ 24 h). `generate_intervention_light_diet`
applies the NHANES clock shift instead, which pushed the 45 windows starting at 0 or 1 h across the day boundary and
split them into two pieces (runs before INTERVENTION_WINDOW_TAG).
"""
intervention_day_diet(policy, lux) = CircadianPOMDP.generate_light_diet_basic(days=1, light_on=policy.start,
    light_off=policy.start + policy.duration, lux_on=lux, lux_off=0.0, sleep_light=false, sleep_dark=missing)

# Compact event storage: only the post-shift CBTmin events, as columns (particle id, time). The shared baseline
# events are saved once per condition (postBaseline_*.jld2), and full event records are rebuilt on load, so a
# reloaded scenario is identical to a freshly simulated one. ~0.5 MB per scenario instead of ~10 MB.
"""Columns (Int16 particle id, Float64 t_hours since the start of the post segment) of post-shift events."""
compact_events(ev) = (Int16[e.particle_id for e in ev], Float64[e.t_hours for e in ev])

"""Rebuild event records (as returned by simulate_particle_population_events_only) from compact columns."""
expand_events(ids, t) = [(particle_id=Int(i), step_idx=round(Int, x / DT_HOURS) + 1, clock_h=mod(x, 24.0), t_hours=x,
                          datetime=START_DATE + Dates.Millisecond(round(Int, x * 3600 * 1000))) for (i, x) in zip(ids, t)]

"""
Run only the post-baseline segment (e.g. 50 days target or 1 day intervention + 50 days target)
from post_baseline_particles, then merge with baseline_cbtmin_events so the full trajectory
has continuous t_hours. If save_path exists, the compact post-shift events are loaded and merged;
otherwise the segment is simulated (events only) and, if save_path is given, saved compactly.
"""
function simulate_post_merge_and_extract_cbtmin(
    post_diet, post_baseline_particles, params,
    baseline_cbtmin_events;
    save_path=nothing,
    baseline_hours::Float64=BASELINE_PST_HOURS
)
    n_particles = length(post_baseline_particles)
    transition_fn = jetlag_transition

    post_events = if save_path !== nothing && isfile(save_path)
        data = load(save_path)
        expand_events(data["post_particle_id"], data["post_t_hours"])
    else
        # Same seed for every scenario: common random numbers across policies and the no-intervention baseline
        ev, _, _ = simulate_particle_population_events_only(
            post_baseline_particles, post_diet, params, transition_fn, START_DATE;
            dt=DT_HOURS, verbose=false, force_threading=RESCUE_MINI, seed=SIMULATION_SEED
        )
        if save_path !== nothing
            mkpath(dirname(save_path))
            post_particle_id, post_t_hours = compact_events(ev)
            num_particles = n_particles
            post_duration_hours = length(post_diet) * DT_HOURS
            write_atomically(save_path) do p
                jldsave(p; post_particle_id, post_t_hours, num_particles, baseline_hours, post_duration_hours,
                        dt_hours=DT_HOURS, format="compact_post_events_v1")
            end
        end
        ev
    end
    merged_events = merge_baseline_and_post_events(baseline_cbtmin_events, post_events, baseline_hours)
    return cbtmin_events_to_trajectory_list(merged_events, n_particles)
end

# ================================================================
# LOAD PARTICLES / PARAMETERS
# ================================================================

println("📊 Loading simulation components...")
all_initial_particles = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]

n_particles = RESCUE_MINI ? N_PARTICLES_MINI : N_PARTICLES_DEFAULT
initial_particles = all_initial_particles[1:n_particles]

params = model_parameters()
# All particles are simulated (indices must match the regular-schedule baseline); excluded ones get no adaptation time.
const INCLUDED_IDS = Set(included_particle_ids())
println("   ✓ Loaded $(n_particles) particles (subset of $(length(all_initial_particles))) and model parameters; " *
        "$(count(<=(n_particles), INCLUDED_IDS)) in the shared analysis set")
if RESCUE_MINI
    println("   [MINI] Reduced run: 50 particles, 2 TZ, 3 policies, 20 d target")
end

intervention_targets = RESCUE_MINI ? TIMEZONE_SHIFTS_MINI : TIMEZONE_SHIFTS_FULL

# ================================================================
# MAIN LOOP OVER LIGHT CONDITIONS
# ================================================================

conditions_to_run = RESCUE_MINI ? light_conditions[1:1] : light_conditions
if !isempty(RESCUE_CONDITION)
    conditions_to_run = filter(c -> c.name == RESCUE_CONDITION, conditions_to_run)
    isempty(conditions_to_run) && error("RESCUE_CONDITION=$(RESCUE_CONDITION) matches no light condition " *
                                        "($(join([c.name for c in light_conditions], ", ")))")
end
for (condition_idx, condition) in enumerate(conditions_to_run)
    println("\n" * "="^80)
    println("🔬 PROCESSING CONDITION $(condition_idx)/$(length(conditions_to_run)): $(condition.description)")
    println("="^80)

    # Post-shift window: per condition (INTERVENTION_TARGET_DAYS in jetlag_reference.jl)
    target_days = RESCUE_MINI ? TARGET_DAYS_MINI : intervention_target_days(condition.name)

    # Save under 4_JetLagIntervention (from jetlag_reference.jl)
    save_dir = intervention_condition_dir(n_particles, condition.name; mini=RESCUE_MINI, target_days=target_days)
    mkpath(PARTICLE_HIST_DIR)
    mkpath(save_dir)
    println("   💾 Saving to: $(save_dir)")
    println("   📂 Particle histories: $(PARTICLE_HIST_DIR)")

    # ------------------------------------------------------------
    # LOAD REGULAR SCHEDULE BASELINE CBTmin INFO (per condition)
    # ------------------------------------------------------------
    println("   📊 Loading regular schedule baseline CBTmin information ($(BASELINE_PST_DAYS) days, $(condition.name))...")
    regular_histories_path = regular_histories_file(condition=condition.name)
    if !isfile(regular_histories_path)
        error("Regular schedule particle histories not found at: $(regular_histories_path)")
    end
    global REGULAR_BASELINE_CBTMINS, REGULAR_BASELINE_COUNTS = load_baseline_steady_state_cbtmin(
        regular_histories_path; baseline_hours=BASELINE_PST_HOURS,
        min_consecutive_days=MIN_CONSECUTIVE_DAYS, threshold=Float64(STEADY_STATE_THRESHOLD))
    n_with_baseline = count(x -> x !== nothing, REGULAR_BASELINE_CBTMINS)
    println("   ✓ Baseline steady-state CBTmin found for $(n_with_baseline) / $(length(REGULAR_BASELINE_CBTMINS)) particles ($(condition.name))")

    # Adaptation target = settled pre-shift CBTmin + shift (day-70 value where no settled value exists); computed
    # once by the baseline stage, so concurrent policy jobs only read it
    settled = settled_preshift_cbtmin(condition.name; compute=RESCUE_STAGE in ("all", "baseline"))
    global TARGET_REFERENCE_CBTMINS = [s === nothing ? b : s for (s, b) in zip(settled, REGULAR_BASELINE_CBTMINS)]
    n_moved = count(i -> settled[i] !== nothing && REGULAR_BASELINE_CBTMINS[i] !== nothing &&
                         abs(mod(settled[i] - REGULAR_BASELINE_CBTMINS[i] + 12, 24) - 12) > EXPECTED_CBTMIN_TOLERANCE, INCLUDED_IDS)
    println("   ✓ Target from the settled pre-shift phase ($(SETTLED_REFERENCE_DAYS) d); differs from the day-70 phase by " *
            "> $(EXPECTED_CBTMIN_TOLERANCE) h for $(n_moved) included particles")

    LD_PST, target_timezones = generate_light_diets_for_condition(condition; baseline_days=BASELINE_PST_DAYS,
                                                                  target_days=target_days)
    println("   ✓ Light diets: first TZ $(length(LD_PST) ÷ (24*60)) d, $(length(target_timezones)) target TZ × $(target_days) d" *
            (RESCUE_MINI ? " [MINI]" : ""))

    # Post-baseline seeding: run the baseline once per condition; baseline and intervention runs
    # use only the post segment (target or intervention+target) from these particles and merge
    # with baseline_cbtmin_events, avoiding re-running the baseline segment for each scenario.
    post_baseline_particles, baseline_cbtmin_events = load_or_compute_post_baseline(
        condition, LD_PST, initial_particles, params
    )

    resume_info = check_resume_capability(save_dir, policy_indices; load_cumulative=(RESCUE_STAGE == "all"))

    # ------------------------------------------------------------
    # BASELINE ANALYSIS (NO INTERVENTION) – STEADY STATE
    # ------------------------------------------------------------
    # Run only target segment from post_baseline_particles, merge with baseline_cbtmin_events.

    if resume_info.can_resume_baseline
        println("   📁 Found existing baseline steady-state results, loading...")
        baseline_results = resume_info.baseline_results
        println("   ✓ Loaded baseline results for $(length(baseline_results)) timezone shifts")
    elseif RESCUE_STAGE in ("policies", "aggregate")
        error("Baseline results not found in $(save_dir)\nRun RESCUE_STAGE=baseline first.")
    else
        println("   🔄 Computing fresh baseline steady-state results...")
        baseline_results = Dict()

        # Sequential over timezones; parallel over particles. (simulate_*_events_only already
        # uses the thread pool for the simulation, so parallelizing over timezones would cause
        # nested contention and we would lose particle-level parallelism without gaining speed.)
        for tz_shift in intervention_targets
            println("     Computing baseline for timezone $(tz_shift)h shift...")

            target_diet = target_timezones[tz_shift]
            baseline_ph_path = joinpath(PARTICLE_HIST_DIR, "postEvents_jetLagIntervention_baseline_$(RESCUE_MINI ? "mini" : string(BASELINE_PST_DAYS + target_days))days_$(condition.name)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_TZ$(tz_shift)h.jld2")

            baseline_cbtmin = simulate_post_merge_and_extract_cbtmin(
                target_diet, post_baseline_particles, params, baseline_cbtmin_events;
                save_path=baseline_ph_path, baseline_hours=BASELINE_PST_HOURS
            )

            n_traj = length(baseline_cbtmin)
            per_particle = Vector{Union{Nothing, NamedTuple}}(undef, n_traj)
            Threads.@threads for i in 1:n_traj
                per_particle[i] = compute_adaptation_time_steady_state(baseline_cbtmin[i], tz_shift; baseline_days=BASELINE_PST_DAYS)
            end

            times_per_particle = Float64[r === nothing ? NaN : r.time for r in per_particle]
            directions_per_particle = [r === nothing ? :invalid : r.direction for r in per_particle]
            valid_times = filter(isfinite, times_per_particle)
            if !isempty(valid_times)
                median_adapt = median(valid_times)
                q25 = quantile(valid_times, 0.25)
                q75 = quantile(valid_times, 0.75)
                ci95 = quantile(valid_times, [0.025, 0.975])
                min_t = minimum(valid_times)
                max_t = maximum(valid_times)
                n_total = count(isfinite, times_per_particle)
                n_forward = count(d -> d == :forward, directions_per_particle)
                n_backward = count(d -> d == :backward, directions_per_particle)
                baseline_results[tz_shift] = (
                    timezone_shift = tz_shift,
                    adaptation_times = times_per_particle,
                    directions = directions_per_particle,
                    median_adaptation = median_adapt,
                    q25_adaptation = q25,
                    q75_adaptation = q75,
                    ci95_lower = ci95[1],
                    ci95_upper = ci95[2],
                    min_adaptation = min_t,
                    max_adaptation = max_t,
                    n_forward = n_forward,
                    n_backward = n_backward,
                    sample_size = n_total
                )
                println("       $(tz_shift)h shift: median=$(round(median_adapt, digits=2)) d [Q25=$(round(q25,digits=2)), Q75=$(round(q75,digits=2))], n=$(n_total)")
                println("         95% CI ≈ [$(round(ci95[1],digits=2)), $(round(ci95[2],digits=2))], min=$(min_t), max=$(max_t), forward=$(n_forward), backward=$(n_backward)")
            else
                println("       ❌ No valid baseline steady-state adaptations for timezone $(tz_shift)h")
            end
        end

        println("   ✓ Baseline steady-state analysis complete for $(condition.description)")
        save_checkpoint(save_dir, "baseline_results_steadyState", baseline_results)
    end
    RESCUE_STAGE == "baseline" && continue

    # ------------------------------------------------------------
    # INTERVENTION ANALYSIS – STEADY STATE
    # ------------------------------------------------------------

    println("\n   🧪 Testing intervention policies (steady-state adaptation)...")

    if resume_info.can_resume_interventions
        println("   📁 Found existing intervention results, loading...")
        intervention_results = resume_info.intervention_results
        completed_policies = length(intervention_results)
        println("   ✓ Loaded results for $(completed_policies) of $(length(policy_indices)) policies")
    else
        println("   🔄 Starting fresh intervention testing...")
        intervention_results = Dict()
        completed_policies = 0
    end
    if RESCUE_STAGE == "aggregate"
        missing_policies = setdiff(policy_indices, keys(intervention_results))
        isempty(missing_policies) ||
            error("$(length(missing_policies)) policies have no results in $(save_dir): $(missing_policies)")
    end

    for policy_idx in policy_indices
        if haskey(intervention_results, policy_idx)
            continue
        end
        policy = intervention_policies[policy_idx]
        t_policy = time()

        println("     📊 Policy $(policy_idx)/$(length(intervention_policies)): start=$(policy.start)h, duration=$(policy.duration)h")

        policy_results = Dict()
        intervention_diet = intervention_day_diet(policy, condition.intervention_lux)

        # Sequential over timezones; parallel over particles. (simulate_*_events_only already
        # uses the thread pool for the simulation, so parallelizing over timezones would cause
        # nested contention and we would lose particle-level parallelism without gaining speed.)
        for tz_shift in intervention_targets
            if !haskey(baseline_results, tz_shift)
                continue
            end

            target_diet = target_timezones[tz_shift]
            baseline = baseline_results[tz_shift]

            post_diet = vcat(intervention_diet, target_diet)
            intervention_ph_path = joinpath(PARTICLE_HIST_DIR, "postEvents_jetLagIntervention_policy$(policy_idx)_$(RESCUE_MINI ? "mini" : string(BASELINE_PST_DAYS + target_days))days_$(condition.name)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_$(INTERVENTION_WINDOW_TAG)_TZ$(tz_shift)h.jld2")

            intervention_cbtmin = simulate_post_merge_and_extract_cbtmin(
                post_diet, post_baseline_particles, params, baseline_cbtmin_events;
                save_path=(SAVE_INTERVENTION_HISTORIES ? intervention_ph_path : nothing), baseline_hours=BASELINE_PST_HOURS
            )

            n_traj = length(intervention_cbtmin)
            per_particle_int = Vector{Union{Nothing, NamedTuple}}(undef, n_traj)
            Threads.@threads for i in 1:n_traj
                per_particle_int[i] = compute_adaptation_time_steady_state(intervention_cbtmin[i], tz_shift; baseline_days=BASELINE_PST_DAYS)
            end

            times_int_full = Float64[r === nothing ? NaN : r.time for r in per_particle_int]
            directions_int = [r === nothing ? :invalid : r.direction for r in per_particle_int]
            valid_int_times = filter(isfinite, times_int_full)
            if !isempty(valid_int_times)
                int_median = median(valid_int_times)
                int_q25 = quantile(valid_int_times, 0.25)
                int_q75 = quantile(valid_int_times, 0.75)
                int_ci95 = quantile(valid_int_times, [0.025, 0.975])
                int_min = minimum(valid_int_times)
                int_max = maximum(valid_int_times)
                n_int = count(isfinite, times_int_full)
                n_forward_int = count(d -> d == :forward, directions_int)
                n_backward_int = count(d -> d == :backward, directions_int)
                baseline_median = baseline.median_adaptation
                adaptation_time_difference = int_median - baseline_median   # negative = faster

                policy_results[tz_shift] = (
                    timezone_shift = tz_shift,
                    baseline_median_adaptation = baseline_median,
                    intervention_median_adaptation = int_median,
                    adaptation_time_difference = adaptation_time_difference,
                    baseline_sample_size = baseline.sample_size,
                    intervention_sample_size = n_int,
                    baseline_times = baseline.adaptation_times,
                    intervention_times = times_int_full,
                    baseline_ci95_lower = baseline.ci95_lower,
                    baseline_ci95_upper = baseline.ci95_upper,
                    baseline_min = baseline.min_adaptation,
                    baseline_max = baseline.max_adaptation,
                    baseline_n_forward = baseline.n_forward,
                    baseline_n_backward = baseline.n_backward,
                    intervention_ci95_lower = int_ci95[1],
                    intervention_ci95_upper = int_ci95[2],
                    intervention_min = int_min,
                    intervention_max = int_max,
                    intervention_n_forward = n_forward_int,
                    intervention_n_backward = n_backward_int
                )
                println("        TZ $(tz_shift)h: baseline=$(round(baseline_median, digits=2)) d, " *
                        "intervention=$(round(int_median, digits=2)) d, " *
                        "Δ=$(round(adaptation_time_difference, digits=2)) d")
            end
        end

        intervention_results[policy_idx] = (
            policy = policy,
            results_by_timezone = policy_results
        )

        save_policy_results_incremental(save_dir, policy_idx, policy_results, policy)
        println("       ⏱ policy $(policy_idx): $(round(time() - t_policy; digits=1)) s"); flush(stdout)

        # Single-process runs only: concurrent policy jobs would overwrite each other's snapshot
        if RESCUE_STAGE == "all" && (policy_idx % 20 == 0 || policy_idx == length(intervention_policies))
            save_checkpoint(save_dir, "intervention_results_cumulative_steadyState", intervention_results)
            println("       💾 Saved cumulative steady-state results through policy $(policy_idx)/$(length(intervention_policies))")
        end
    end
    if RESCUE_STAGE == "policies"
        println("   ✅ Policies $(first(policy_indices))–$(last(policy_indices)) done for $(condition.name); " *
                "run RESCUE_STAGE=aggregate once every policy has finished")
        continue
    end

    # ------------------------------------------------------------
    # AGGREGATE POLICY PERFORMANCE (BASED ON ADAPTATION TIME)
    # ------------------------------------------------------------

    println("\n   📊 Analyzing intervention effectiveness (steady-state adaptation)...")

    best_policies_by_timezone = Dict()

    for tz_shift in intervention_targets
        policy_performances = []

        for (policy_idx, policy_data) in intervention_results
            if haskey(policy_data.results_by_timezone, tz_shift)
                result = policy_data.results_by_timezone[tz_shift]
                push!(policy_performances, (
                    policy_idx = policy_idx,
                    policy = policy_data.policy,
                    adaptation_time_difference = result.adaptation_time_difference,  # negative = better
                    baseline_median_adaptation = result.baseline_median_adaptation,
                    intervention_median_adaptation = result.intervention_median_adaptation,
                    baseline_sample_size = result.baseline_sample_size,
                    intervention_sample_size = result.intervention_sample_size
                ))
            end
        end

        sort!(policy_performances, by = x -> x.adaptation_time_difference, rev=false)
        best_policies_by_timezone[tz_shift] = policy_performances
    end

    overall_policy_performance = Dict()

    for (policy_idx, policy_data) in intervention_results
        policy = policy_data.policy
        diffs = Float64[]

        for tz_shift in intervention_targets
            if haskey(policy_data.results_by_timezone, tz_shift)
                result = policy_data.results_by_timezone[tz_shift]
                push!(diffs, result.adaptation_time_difference)
            end
        end

        if !isempty(diffs)
            avg_diff = mean(diffs)
            overall_policy_performance[policy_idx] = (
                policy = policy,
                avg_adaptation_time_difference = avg_diff,
                differences = diffs,
                n_timezones = length(diffs)
            )
        end
    end

    # ------------------------------------------------------------
    # SAVE RESULTS (STEADY-STATE VERSION)
    # ------------------------------------------------------------

    println("\n   💾 Saving systematic intervention (steady-state) results for $(condition.description)...")

    final_results = Dict(
        "analysis_info" => Dict(
            "description" => "Systematic jet lag rescue analysis using light timing interventions – steady-state adaptation time",
            "light_condition" => condition,
            "intervention_policies" => intervention_policies,
            "intervention_targets" => intervention_targets,
            "analysis_date" => string(now()),
            "total_particles" => length(initial_particles),
            "methodology" => "Steady-state detection of CBTmin in the new timezone; adaptation time = first day meeting stability criteria",
            "target_days" => target_days,
            "target_reference" => "settled pre-shift CBTmin after $(SETTLED_REFERENCE_DAYS) d ($(TARGET_REFERENCE_TAG))",
            "intervention_window" => INTERVENTION_WINDOW_TAG
        ),
        "baseline_results" => baseline_results,
        "intervention_results" => intervention_results,
        "best_policies_by_timezone" => best_policies_by_timezone,
        "overall_policy_performance" => overall_policy_performance
    )

    write_atomically(p -> jldsave(p; final_results), joinpath(save_dir, "systematic_rescue_analysis_steadyState.jld2"))
    println("   ✓ Results saved to $(save_dir)/systematic_rescue_analysis_steadyState.jld2")

    # Quick summary
    println("\n   📈 CONDITION SUMMARY (steady-state): $(condition.description)")
    total_policies = length(intervention_policies)
    completed_policies = length(intervention_results)
    success_rate = total_policies > 0 ? round(completed_policies / total_policies * 100, digits=1) : 0.0
    println("   • Completed $(completed_policies)/$(total_policies) policies ($(success_rate)%)")

    if !isempty(overall_policy_performance)
        sorted_policies = sort(collect(overall_policy_performance),
                               by = x -> x[2].avg_adaptation_time_difference,
                               rev=false)
        if !isempty(sorted_policies)
            best_policy = sorted_policies[1][2].policy
            best_avg_diff = sorted_policies[1][2].avg_adaptation_time_difference
            println("   • Best policy (by avg adaptation time): $(best_policy.start)h-$(best_policy.start + best_policy.duration)h")
            println("     Mean Δ adaptation time (intervention - baseline): $(round(best_avg_diff, digits=2)) days")
        end
    end

    println("   ✅ $(condition.description) steady-state policy analysis complete!")
end

println("\n" * "="^80)
println("🎯 UNIFIED JET LAG RESCUE ANALYSIS (STEADY STATE) COMPLETE")
println("="^80)



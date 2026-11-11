"""
# CircadianPOMDP

Circadian rhythm and sleep-wake model for particle-based simulation and analysis, with
support for **jet lag experiments**: baseline runs, timezone shifts, light interventions,
and adaptation-time estimation.

## Quick start (JetLag pipeline)

```julia
using CircadianPOMDP
using Dates

# 1. Model parameters and initial particles (e.g. for jet lag: sampled params, single location)
params = model_parameters()
particles = initialize_particles(1000, DateTime(2023, 1, 1); state_params_mode=:sampled_normal)

# 2. Build light diet: precomputed (Vector{CircadianAction}) or on-the-fly (CircadianActionExt)
#    Precomputed (e.g. baseline PST): use generate_light_diet_basic or generate_light_diets_for_condition
LD = generate_light_diet_basic(days=35, lux_on=150.0,
    light_on=REGULAR_SCHEDULE_LIGHT_ON_H, light_off=REGULAR_SCHEDULE_LIGHT_OFF_H,
    lux_off=0.0, sleep_light=missing, sleep_dark=missing)

# 3. Simulate: use circadian_transition_sleep for precomputed light, circadian_transition_ext for on-the-fly light
#    Memory-efficient (events only): simulate_particle_population_events_only
cbtmin_events, onset_ev, offset_ev = simulate_particle_population_events_only(
    particles, LD, params, circadian_transition_sleep, DateTime(2023, 1, 1); dt=1/60, verbose=false)

# 4. Analysis: baseline CBTmin, merge baseline+post, MLE summaries from events
base_traj, base_counts = baseline_cbtmin_from_events(cbtmin_events, length(particles), 35*24)
summaries = compute_mle_population_summaries(
    cbtmin_events, onset_ev, offset_ev, particles; verbose=false)
```

## API overview

| Domain | Key functions | Use when |
|--------|----------------|----------|
| **Initialization** | `initialize_particles(N, t0; state_params_mode=:sampled_normal)` | JetLag: one ensemble, sampled params |
| **Transitions** | `transition`, `circadian_transition_sleep`, `circadian_transition_ext` | Precomputed light → sleep; on-the-fly diet → ext |
| **Light diets** | `generate_light_diet_basic`, `generate_light_diets_for_condition`, `build_baseline_diet_full`, `build_intervention_diet_full` | Baseline PST, intervention sweeps |
| **Simulation** | `simulate_particle_from_initial_population`, `simulate_particle_population_events_only` | Full histories vs events-only (recommended for long runs) |
| **Events / metrics** | `baseline_cbtmin_from_events`, `merge_baseline_and_post_events`, `cbtmin_events_to_trajectory_list` | Downstream adaptation and MLE |
| **MLE summaries** | `compute_mle_population_summaries`, `compute_mle_population_summaries_from_events` | Population CBTmin/sleep summaries |

For [`transition`](@ref), `CircadianAction` and `CircadianActionExt` share the same default `sleep_light_fraction` ([`EYELID_LIGHT_FRACTION`](@ref)). Pass `sleep_light_fraction=1.0` or `0.0` if you need no attenuation or full block during sleep.
"""
module CircadianPOMDP

# Import required packages (JetLag-relevant only)
using Distributions
using Statistics
using JLD2
using Plots
using Dates   # DateTime in light diet and initialization
using Random
using StableRNGs

# Include source files in dependency order (JetLag pipeline only)
include("pomdp/circadian_model.jl")
include("simulation/generate_light_diet_basic.jl")
include("simulation/light_intensity_core.jl")
include("simulation/light_diet_functions.jl")
include("pomdp/circadian_transition.jl")
include("simulation/sleep_CBTmin_detection.jl")
include("pomdp/initialization_functions.jl")
include("simulation/simulate_particle_population.jl")
include("analysis/particle_metadata.jl")
include("analysis/particle_summaries.jl")
include("analysis/particle_summaries_mle.jl")
include("analysis/steady_state.jl")

# =============================================================================
# Exports (grouped by API domain)
# =============================================================================

# --- Abstract types (use these for API boundaries and dispatch) ---
export AbstractCircadianState, AbstractCircadianAction, AbstractCircadianObservation

# --- Concrete POMDP types ---
export CircadianState, CircadianAction, CircadianActionExt, CircadianObservation
export SleepPeriod

# --- Model parameters ---
export model_parameters, get_space_types, get_all_parameters_flat, print_parameter_summary
export default_circadian_state, default_circadian_action, default_circadian_observation

# --- Transition API ---
# Use `transition` for dispatch; `circadian_transition_sleep` (precomputed light) or
# `circadian_transition_ext` (on-the-fly light) for explicit mode selection.
export transition, circadian_transition_sleep, circadian_transition_ext
export is_within_action_window, get_action_duration_hours, get_action_num_steps
export create_action_sequence, get_current_action

# --- Eyelid / sleep light (Process L) ---
export EYELID_LIGHT_FRACTION, OCULAR_LIGHT_NOISE_REL_SD

# --- Sleep / CBTmin detection ---
export get_CBTmin, get_sleep_periods

# --- Initialization ---
export initialize_particles, DEFAULT_INITIAL_SLEEP_PROB_ASLEEP

# --- Light diets (JetLag: array-based and Regular_V4 on-the-fly) ---
export generate_light_diet_basic
export REGULAR_SCHEDULE_LIGHT_ON_H, REGULAR_SCHEDULE_LIGHT_OFF_H
export REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H
export generate_shifted_dark_diet, generate_light_period_diet, generate_intervention_light_diet
export get_intervention_policies, generate_light_diets_for_condition
export build_baseline_diet_full, build_intervention_diet_full
export JETLAG_MEAN_ALIGNMENT_SHIFT_H
export get_light_intensity, get_light_and_sleep, compute_light_from_action
export get_timezone_center_lon, infer_timezone_center_from_longitude
export get_available_light_diets
export LIGHT_DIETS

# --- Simulation ---
export simulate_particle_population
export simulate_particle_from_initial_population
export simulate_particle_population_events_only

# --- Particle event utilities ---
export baseline_cbtmin_from_events, mean_sleep_midpoint_clock_h_from_events, circular_mean_clock_h
export load_particle_histories, load_particle_summaries, extract_summary_features
export events_to_trajectories, cbtmin_events_to_trajectory_list
export cbtmin_events_from_state_histories, merge_baseline_and_post_events
export build_metrics_from_events

# --- Summaries (MLE) ---
export plot_summary_histograms
export compute_mle_particle_summary, compute_mle_population_summaries
export analyze_mle_summaries, print_mle_summary_stats
export save_mle_summaries, load_mle_summaries

# --- Steady-state detection and 24-hour cyclic utilities ---
export unwrap24, wrap24
export load_baseline_steady_state_cbtmin
export is_steady_state_window, find_baseline_steady_state_cbtmin
export find_first_steady_state_day, compute_adaptation_time
export normalize_time!, normalize_time

end

# Generate Particle Histories for regular (non–jet lag) schedule
# Uses CircadianState, CircadianActionExt (on-the-fly light), and regular_schedule_transition
# (circadian_transition_ext, stochastic light, SIMULATION_SEED; see revision_baseline.jl). Length: BASELINE_PST_DAYS.
#
# Uses simulate_particle_population_events_only for fast, memory-efficient simulation
# (O(events) memory instead of O(N×T), particle-parallel threading, sub-step CBTmin precision).
#
# Paths and constants from jetlag_reference.jl (Data/JetLag/2_RegularScheduleAnalysis/particleHistories/).

using Random
using Dates
using JLD2
using CircadianPOMDP

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

Random.seed!(SIMULATION_SEED)

# Check threading status
println("Threading Status:")
println("  Available threads: $(Threads.nthreads())")
if Threads.nthreads() == 1
    println("  ⚠️  Running in single-threaded mode")
    println("  💡 For better performance, start Julia with: julia -t auto your_script.jl")
else
    println("  ✅ Multi-threaded mode active")
end
println()

# --- Paths (from jetlag_reference.jl) ---
ensure_jetlag_data_dirs()
particles_path = initial_particles_path(N_PARTICLES_DEFAULT)

# --- Simulation parameters (from jetlag_reference.jl) ---
start_date = START_DATE
duration_days = BASELINE_PST_DAYS
dt_hours = DT_HOURS

params = model_parameters()

# --- Load particles (CircadianState includes location and time) ---
println("Loading initial particles...")
particles = load(particles_path)["initial_particles"]
println("  Loaded $(length(particles)) particles")

# --- Run for each light diet condition (HighLight10000lux, LowLight150lux) ---
for condition in REGULAR_SCHEDULE_CONDITIONS
    light_diet = regular_schedule_light_diet_for_condition(condition)
    save_dir = regular_schedule_particle_hist_dir(; condition)
    mkpath(save_dir)

    println()
    println("="^60)
    println("Light diet: $condition ($light_diet)")
    println("="^60)

    # --- Single action spanning the run; light is computed from state.t each step ---
    action = CircadianActionExt(
        light_diet, REGULAR_SCHEDULE_SLEEP_DIET, REGULAR_SCHEDULE_SOLAR_MODEL,
        dt_hours, start_date, start_date + Day(duration_days)
    )
    actions = fill(action, get_action_num_steps(action))
    simulation_duration = length(actions) * dt_hours
    println("Simulating ($(length(particles)) particles, $(length(actions)) steps) [on-the-fly events]...")

    # --- Simulate (on-the-fly stochastic light, sleep blocks light, full dynamics) ---
    transition_fn = regular_schedule_transition
    cbtmin_events, sleep_onset_events, sleep_offset_events = simulate_particle_population_events_only(
        particles, actions, params, transition_fn, start_date;
        dt=dt_hours, verbose=REGULAR_SCHEDULE_VERBOSE, seed=SIMULATION_SEED
    )

    # --- Build metrics from events ---
    println("Building circadian and sleep metrics from events...")
    circadian_metrics_per_particle, sleep_metrics_per_particle = build_metrics_from_events(
        cbtmin_events, sleep_onset_events, sleep_offset_events, length(particles);
        simulation_duration=simulation_duration
    )

    # --- Save (events-based format for downstream scripts; filename from jetlag_reference.jl) ---
    out_path = regular_histories_file(; condition)
    println("Saving to $out_path")
    @save out_path cbtmin_events sleep_onset_events sleep_offset_events circadian_metrics_per_particle sleep_metrics_per_particle particles dt_hours start_date simulation_duration
    println("Done for $condition.")
end
println()
println("All light diet conditions complete.")

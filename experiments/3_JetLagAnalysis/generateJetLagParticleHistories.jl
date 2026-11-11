# Generate Particle Histories for jet lag light diets
# Uses CircadianState, CircadianAction (from precomputed light diet), and jetlag_transition (stochastic light).
# Every shift uses SIMULATION_SEED, so the BASELINE_PST_DAYS baseline is identical across shifts and identical to
# the regular-schedule run (same per-particle noise). All 1,000 particles are simulated so indices match the
# regular-schedule histories; analyses keep only included_particle_ids().
#
# Builds light diets via generate_light_diets_for_condition (same logic as rescue pipeline). Post-shift window per
# condition (INTERVENTION_TARGET_DAYS: 150 lux 100 d, 10,000 lux 50 d); files that already exist for that window
# are kept (same seed, so a rerun would reproduce them).
# Uses simulate_particle_population_events_only for fast, memory-efficient simulation.
#
# Paths and constants from jetlag_reference.jl (Data/JetLag/3_JetLagAnalysis/particleHistories/).

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
dt_hours = DT_HOURS

params = model_parameters()
transition_fn = jetlag_transition

# --- Load particles (shared across conditions) ---
println("Loading initial particles...")
particles = load(particles_path)["initial_particles"]
println("  Loaded $(length(particles)) particles")
println()

# --- Loop over light diet conditions (HighLight10000lux, LowLight150lux) ---
for condition in light_conditions_specs()
    cond_name = condition.name
    println("="^60)
    println("Light diet condition: $cond_name")
    println("="^60)

    save_dir = jetlag_analysis_particle_hist_dir(condition=cond_name)
    mkpath(save_dir)

    # Build jet lag light diets for this condition
    total_days = jetlag_total_days(cond_name)
    LD_PST, target_timezones = generate_light_diets_for_condition(condition; baseline_days=BASELINE_PST_DAYS,
                                                                  target_days=intervention_target_days(cond_name))
    LD_JL = [vcat(LD_PST, target_timezones[i]) for i in 1:length(target_timezones)]
    println("Built $(length(LD_JL)) jet lag light diet scenarios")
    println()

    # Simulate each scenario
    for (i, LD) in enumerate(LD_JL)
        out_path = particle_hist_filename_for_shift(i; condition=cond_name, total_days=total_days)
        if isfile(out_path)
            println("Scenario $i: $(basename(out_path)) exists, keeping it")
            continue
        end
        simulation_duration = length(LD) * dt_hours
        println("Simulating scenario $i of $(length(LD_JL)) ($(length(particles)) particles, $(length(LD)) steps)...")

        cbtmin_events, sleep_onset_events, sleep_offset_events = simulate_particle_population_events_only(
            particles, LD, params, transition_fn, start_date;
            dt=dt_hours, verbose=JETLAG_VERBOSE, seed=SIMULATION_SEED
        )

        println("Building circadian and sleep metrics from events...")
        circadian_metrics_per_particle, sleep_metrics_per_particle = build_metrics_from_events(
            cbtmin_events, sleep_onset_events, sleep_offset_events, length(particles);
            simulation_duration=simulation_duration
        )

        println("Saving to $out_path")
        @save out_path cbtmin_events sleep_onset_events sleep_offset_events circadian_metrics_per_particle sleep_metrics_per_particle particles dt_hours start_date simulation_duration
        println("Completed scenario $i/$(length(LD_JL))")
    end

    println("✓ Condition $cond_name complete.")
    println()
end

println("Done.")

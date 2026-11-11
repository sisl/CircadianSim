#!/usr/bin/env julia
# Run from repo root:  julia --project=. quickstart/02_minimal_jetlag.jl
# No setup or saved data required. Verifies install with a short simulation (3 days, 50 particles).

using CircadianPOMDP
using Dates

println("CircadianPOMDP — minimal example (3 days, 50 particles)")
println("=======================================================")

# --- 1. Model parameters and particle initialization ---
t0 = DateTime(2023, 1, 1)
params = model_parameters()
particles = initialize_particles(50, t0; state_params_mode=:sampled_normal)

# --- 2. Generate a simple light diet (3 days; same clock as Regular_V4 / jet-lag arrays) ---
LD = generate_light_diet_basic(days=3, lux_on=150.0,
    light_on=REGULAR_SCHEDULE_LIGHT_ON_H, light_off=REGULAR_SCHEDULE_LIGHT_OFF_H,
    lux_off=0.0, sleep_light=missing, sleep_dark=missing)

# --- 3. Simulate (events-only, memory-efficient) ---
println("Simulating (events-only)...")
cbtmin_events, onset_ev, offset_ev = simulate_particle_population_events_only(
    particles, LD, params, circadian_transition_sleep, t0; dt=1/60, verbose=false)

# --- 4. Baseline CBTmin extraction ---
n_hrs = 3 * 24
base_traj, base_counts = baseline_cbtmin_from_events(cbtmin_events, length(particles), n_hrs)
println("Baseline CBTmin counts per particle: min=$(minimum(base_counts)), max=$(maximum(base_counts))")

# --- 5. MLE population summaries ---
summaries = compute_mle_population_summaries(
    cbtmin_events, onset_ev, offset_ev, particles;
    verbose=false,
    start_day=0.0,           # use full 3-day window (default 20.333 is for ~35-day runs)
    min_sleep_episodes=2)     # need only 2 for short run (default 13 is for paper pipeline)
n_ok = count(s -> !isnothing(s.CBTmin_mle), summaries)
println("MLE summaries: $(length(summaries)) particles; CBTmin_mle present for $n_ok")

# --- 6. Steady-state utilities (from package) ---
# unwrap24 / wrap24 handle 24-hour cyclic data
demo = unwrap24([23.0, 23.5, 0.5, 1.0])
println("unwrap24 demo: [23.0, 23.5, 0.5, 1.0] → $(round.(demo, digits=1))")
println("wrap24(25.5) = $(wrap24(25.5))")

# find_baseline_steady_state_cbtmin detects when a particle has stabilized
if any(c -> c >= 3, base_counts)
    idx = findfirst(c -> c >= 3, base_counts)
    ss = find_baseline_steady_state_cbtmin(base_traj[idx]; min_consecutive_days=2)
    println("Particle $idx baseline steady-state CBTmin: $(ss === nothing ? "not found (too short)" : round(ss, digits=2))")
end

# --- 7. Saving / loading (JLD2, used in experiments) ---
println("\nTo save and reload results in the full pipeline, use JLD2:")
println("  using JLD2")
println("  @save \"output.jld2\" cbtmin_events summaries")
println("  data = load(\"output.jld2\")")

println("\nDone. See experiments/ for the full paper pipeline (README and jetlag_reference.jl).")

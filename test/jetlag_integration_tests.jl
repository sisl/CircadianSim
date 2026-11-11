# JetLag experiment pipeline integration tests
# Covers: initialize_particles (sampled), light diets, events-only simulation,
# circadian_transition_sleep / circadian_transition_ext, baseline_cbtmin_from_events,
# merge_baseline_and_post_events, MLE from events, and related APIs.
using Random
using Dates
using Test

@testset "jetlag_integration" begin
    Random.seed!(42)
    params = CircadianPOMDP.model_parameters()
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    dt_h = 1/60

    # --- 1. Initial particles (JetLag style: sampled, single location) ---
    N = 30
    particles = CircadianPOMDP.initialize_particles(N, t0; state_params_mode=:sampled_normal)
    @test length(particles) == N
    @test all(p -> p isa CircadianPOMDP.CircadianState, particles)
    @test all(p -> isapprox(p.x^2 + p.xc^2, 1.0, atol=1e-9), particles)

    # --- 2. Precomputed light diet (CircadianAction) — baseline PST style ---
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=2,
        lux_on=10_000.0, light_on=CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H, light_off=CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_OFF_H,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    @test length(LD) == 2 * 24 * 60
    @test all(a -> a isa CircadianPOMDP.CircadianAction, LD)

    # --- 3. Events-only simulation with circadian_transition_sleep (precomputed diet) ---
    cbtmin_ev, onset_ev, offset_ev = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0;
        dt=dt_h, verbose=false, seed=42)
    @test cbtmin_ev isa Vector
    @test onset_ev isa Vector
    @test offset_ev isa Vector
    @test all(e -> 1 <= e.particle_id <= N, cbtmin_ev)
    sim_dur = length(LD) * dt_h

    # --- 4. Baseline CBTmin from events ---
    baseline_hours = 24.0
    base_traj, base_counts = CircadianPOMDP.baseline_cbtmin_from_events(cbtmin_ev, N, baseline_hours)
    @test length(base_traj) == N
    @test length(base_counts) == N
    @test all(c -> c >= 0, base_counts)

    # --- 6. MLE summaries from events ---
    summaries_ev = CircadianPOMDP.compute_mle_population_summaries(
        cbtmin_ev, onset_ev, offset_ev, particles; verbose=false, start_day=0.5, min_sleep_episodes=2)
    if !isempty(summaries_ev)
        @test all(s -> hasproperty(s, :CBTmin_mle) && hasproperty(s, :tau), summaries_ev)
        analysis = CircadianPOMDP.analyze_mle_summaries(summaries_ev)
        @test analysis isa Dict
        @test haskey(analysis, "population_size")
    end

    # --- 7. CircadianActionExt path (on-the-fly light) — Regular schedule style ---
    action_ext = CircadianPOMDP.CircadianActionExt(
        "Regular_V4_061hEarlier", "open", missing,
        dt_h, t0, t0 + Day(2)
    )
    n_steps = CircadianPOMDP.get_action_num_steps(action_ext)
    actions_ext = fill(action_ext, n_steps)
    transition_ext_det = (s, a, p; rng) -> CircadianPOMDP.circadian_transition_ext(s, a, p; rng, stochastic_light=false)
    cbtmin_ext, onset_ext, offset_ext = CircadianPOMDP.simulate_particle_population_events_only(
        particles, actions_ext, params, transition_ext_det, t0;
        dt=dt_h, verbose=false, seed=42)
    @test cbtmin_ext isa Vector
    @test all(e -> 1 <= e.particle_id <= N, cbtmin_ext)

    # --- 8. transition() API with both action types ---
    s = particles[1]
    a_pre = CircadianPOMDP.CircadianAction(100.0, missing, dt_h)
    s_next_pre = CircadianPOMDP.transition(s, a_pre, params)
    @test s_next_pre isa CircadianPOMDP.CircadianState
    a_ext_single = CircadianPOMDP.CircadianActionExt(
        "Regular_V4_061hEarlier", "open", missing, dt_h, t0, t0 + Hour(1))
    s_next_ext = CircadianPOMDP.transition(s, a_ext_single, params; deterministic_light=true)
    @test s_next_ext isa CircadianPOMDP.CircadianState

    # --- 9. merge_baseline_and_post_events ---
    post_events = [e for e in cbtmin_ev if e.t_hours > baseline_hours]
    merged = CircadianPOMDP.merge_baseline_and_post_events(
        [e for e in cbtmin_ev if e.t_hours <= baseline_hours],
        post_events,
        baseline_hours)
    @test merged isa Vector
    @test all(e -> hasproperty(e, :t_hours) && hasproperty(e, :particle_id), merged)

    # --- 10. cbtmin_events_to_trajectory_list (used by rescue pipeline) ---
    traj_list = CircadianPOMDP.cbtmin_events_to_trajectory_list(cbtmin_ev, N)
    @test traj_list isa Vector
    for item in traj_list
        @test hasproperty(item, :particle_id)
        @test hasproperty(item, :cbtmin_circadian)
        @test hasproperty(item, :cbtmin_absolute)
    end

    # --- 11. generate_light_diets_for_condition / build_baseline_diet_full / build_intervention_diet_full ---
    condition = (baseline_lux=10000.0, target_lux=10000.0, intervention_lux=10000.0)
    LD_PST, targets = CircadianPOMDP.generate_light_diets_for_condition(
        condition; baseline_days=1, target_days=1)
    @test length(LD_PST) == 24 * 60
    full_baseline = CircadianPOMDP.build_baseline_diet_full(condition, 1; baseline_days=1, target_days=1)
    @test length(full_baseline) == length(LD_PST) + length(targets[1])
    policy = (start=8.0, duration=2.0)
    full_interv = CircadianPOMDP.build_intervention_diet_full(
        condition, policy, 1; baseline_days=1, target_days=1)
    @test length(full_interv) == length(LD_PST) + 24*60 + length(targets[1])

    # --- 12. circular_mean_clock_h and mean_sleep_midpoint_clock_h_from_events ---
    hours = [3.0, 4.0, 5.0]
    cm = CircadianPOMDP.circular_mean_clock_h(hours)
    @test cm isa Float64
    @test 0 <= cm < 24 || isnan(cm)
    midpoints = CircadianPOMDP.mean_sleep_midpoint_clock_h_from_events(
        onset_ev, offset_ev, N; t_start_hours=0.0, t_end_hours=sim_dur)
    @test length(midpoints) == N
end

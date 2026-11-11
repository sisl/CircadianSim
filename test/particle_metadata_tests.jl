using Random
using Dates

@testset "particle_event_utilities" begin
    Random.seed!(456)
    params = CircadianPOMDP.model_parameters()
    particles = CircadianPOMDP.initialize_particles(5, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=3,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    dt = 1/60
    cbtmin_ev, onset_ev, offset_ev = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=dt)

    # baseline_cbtmin_from_events returns (trajectories, counts)
    base_traj, base_counts = CircadianPOMDP.baseline_cbtmin_from_events(cbtmin_ev, 5, 1.0)
    @test base_traj isa Vector
    @test base_counts isa Vector
    @test length(base_traj) == 5

    # events_to_trajectories
    traj = CircadianPOMDP.events_to_trajectories(cbtmin_ev, 5)
    @test length(traj) == 5

    # cbtmin_events_to_trajectory_list
    traj_list = CircadianPOMDP.cbtmin_events_to_trajectory_list(cbtmin_ev, 5)
    @test traj_list isa Vector

    # circular_mean_clock_h
    @test CircadianPOMDP.circular_mean_clock_h([1.0, 23.0]) ≈ 0.0 atol=1e-10
    @test isnan(CircadianPOMDP.circular_mean_clock_h(Float64[]))

    # mean_sleep_midpoint_clock_h_from_events
    midpoints = CircadianPOMDP.mean_sleep_midpoint_clock_h_from_events(onset_ev, offset_ev, 5)
    @test length(midpoints) == 5

    # merge_baseline_and_post_events
    merged = CircadianPOMDP.merge_baseline_and_post_events(cbtmin_ev, cbtmin_ev, 72.0)
    @test length(merged) == 2 * length(cbtmin_ev)

    # cbtmin_events_from_state_histories
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)
    ev_from_hist = CircadianPOMDP.cbtmin_events_from_state_histories(PH, dt)
    @test ev_from_hist isa Vector
    @test !isempty(ev_from_hist)
    @test all(e -> haskey(e, :particle_id) || hasproperty(e, :particle_id), ev_from_hist)
    @test all(e -> 0.0 <= e.clock_h < 24.0, ev_from_hist)
    @test all(e -> e.t_hours >= 0.0, ev_from_hist)
end

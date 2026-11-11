using Random
using Dates

@testset "simulation" begin
    Random.seed!(123)
    params = CircadianPOMDP.model_parameters()
    N = 10
    particles = CircadianPOMDP.initialize_particles(N, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=2,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )

    # simulate_particle_from_initial_population
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)
    @test length(PH) == N
    @test all(h -> length(h) == length(LD) + 1, PH)
    @test PH[1][1] == particles[1]
    @test all(h -> h[1] isa CircadianPOMDP.CircadianState, PH)

    # simulate_particle_population (generates its own particles)
    PH2 = CircadianPOMDP.simulate_particle_population(N, LD, params;
        transition_fn=CircadianPOMDP.circadian_transition_sleep)
    @test length(PH2) == N
    @test all(h -> length(h) == length(LD) + 1, PH2)

    # simulate_particle_population_events_only (events-based, no full history)
    using Dates
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    dt_h = 1/60
    cbtmin_ev, onset_ev, offset_ev = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=dt_h, verbose=false)
    @test cbtmin_ev isa Vector
    @test onset_ev isa Vector
    @test offset_ev isa Vector
    @test all(e -> 1 <= e.particle_id <= N, cbtmin_ev)
    @test all(e -> 1 <= e.particle_id <= N, onset_ev)
    @test all(e -> 1 <= e.particle_id <= N, offset_ev)

    # Edge case: single-step simulation (1 particle, 1 action)
    p_single = CircadianPOMDP.initialize_particles(1, DateTime(2023, 1, 1))
    LD_one = LD[1:1]
    PH_single = CircadianPOMDP.simulate_particle_from_initial_population(
        p_single, LD_one, params, CircadianPOMDP.circadian_transition_sleep)
    @test length(PH_single) == 1
    @test length(PH_single[1]) == 2

    # Seeded simulation: two runs with same seed produce identical results
    PH_a = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep; seed=42)
    PH_b = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep; seed=42)
    @test all(PH_a[i][end].x == PH_b[i][end].x for i in 1:N)
    @test all(PH_a[i][end].H == PH_b[i][end].H for i in 1:N)

    # Deterministic prescribed light: different seeds still yield identical trajectories (noise was only in illuminance)
    PH_c = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep; seed=99)
    @test all(PH_a[i][end].x == PH_c[i][end].x for i in 1:N)

    # Seeded events-only: two runs produce identical CBTmin events
    ev_a = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0;
        dt=dt_h, verbose=false, seed=42)
    ev_b = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0;
        dt=dt_h, verbose=false, seed=42)
    @test length(ev_a[1]) == length(ev_b[1])
    @test all(ev_a[1][i].t_hours == ev_b[1][i].t_hours for i in eachindex(ev_a[1]))
end

@testset "simulate_particle_population_events_only: return_final_states" begin
    params = model_parameters()
    t0 = DateTime(2023, 1, 1)
    P = CircadianPOMDP.initialize_particles(3, t0; state_params_mode=:sampled_normal, seed=7)
    LD = generate_light_diet_basic(days=3, lux_on=150.0, light_on=7.0, light_off=23.0,
                                   lux_off=0.0, sleep_light=missing, sleep_dark=missing)
    ev = simulate_particle_population_events_only(P, LD, params, circadian_transition_sleep, t0;
                                                  dt=1/60, verbose=false, use_threading=false, seed=1)
    ev_fs = simulate_particle_population_events_only(P, LD, params, circadian_transition_sleep, t0;
                                                     dt=1/60, verbose=false, use_threading=false, seed=1,
                                                     return_final_states=true)
    @test length(ev_fs) == 4
    @test ev_fs[1:3] == ev                       # events unchanged by the option
    hist = simulate_particle_from_initial_population(P, LD, params, circadian_transition_sleep; seed=1)
    @test all(ev_fs[4][i] == hist[i][end] for i in eachindex(P))   # final state = last state of full history
    # Continuing from the final states reproduces one longer run (deterministic light)
    d1 = LD[1:2*1440]; d2 = LD[2*1440+1:end]
    _, _, _, mid = simulate_particle_population_events_only(P, d1, params, circadian_transition_sleep, t0;
        dt=1/60, verbose=false, use_threading=false, seed=1, return_final_states=true)
    _, _, _, fin = simulate_particle_population_events_only(mid, d2, params, circadian_transition_sleep, t0;
        dt=1/60, verbose=false, use_threading=false, seed=1, return_final_states=true)
    @test all(fin[i] == ev_fs[4][i] for i in eachindex(P))
end

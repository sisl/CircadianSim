using Random
using Dates
using JLD2

@testset "particle_summaries_mle" begin
    Random.seed!(789)
    params = CircadianPOMDP.model_parameters()
    particles = CircadianPOMDP.initialize_particles(20, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=7,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    cbtmin_ev, onset_ev, offset_ev = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=1/60, verbose=false)

    # compute_mle_population_summaries (relaxed min_sleep_episodes for short run)
    summaries = CircadianPOMDP.compute_mle_population_summaries(
        cbtmin_ev, onset_ev, offset_ev, particles; verbose=false, start_day=2.0, min_sleep_episodes=3)
    @test !isempty(summaries)
    s = summaries[1]
    @test hasproperty(s, :tau)
    @test hasproperty(s, :K)
    @test hasproperty(s, :CBTmin_mle)
    @test hasproperty(s, :sleep_onset_mle)
    @test hasproperty(s, :sleep_duration_mle)
    @test s.tau > 20 && s.tau < 28

    # analyze_mle_summaries
    analysis = CircadianPOMDP.analyze_mle_summaries(summaries)
    @test analysis isa Dict

    # print_mle_summary_stats runs without error
    @test nothing === CircadianPOMDP.print_mle_summary_stats(analysis)

    # save_mle_summaries / load_mle_summaries
    path = joinpath(@__DIR__, "output", "test_summaries.jld2")
    mkpath(dirname(path))
    CircadianPOMDP.save_mle_summaries(summaries, path; include_analysis=false, verbose=false)
    @test isfile(path)
    loaded = CircadianPOMDP.load_mle_summaries(path)
    @test length(loaded.summaries) == length(summaries)

    # compute_mle_particle_summary (direct unit test)
    dt = 1/60
    single_summary = CircadianPOMDP.compute_mle_particle_summary(
        cbtmin_ev, onset_ev, offset_ev, 1, particles[1]; start_day=2.0, min_sleep_episodes=2)
    if single_summary !== nothing
        @test hasproperty(single_summary, :CBTmin_mle)
        @test hasproperty(single_summary, :tau)
    end

    # Edge case: insufficient events returns nothing
    short_p = CircadianPOMDP.initialize_particles(1, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD_short = CircadianPOMDP.generate_light_diet_basic(
        days=1, lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    cbtmin_s, onset_s, offset_s = CircadianPOMDP.simulate_particle_population_events_only(
        short_p, LD_short, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=1/60, verbose=false)
    result = CircadianPOMDP.compute_mle_particle_summary(
        cbtmin_s, onset_s, offset_s, 1, short_p[1]; start_day=20.0, min_sleep_episodes=2)
    @test result === nothing  # start_day=20d beyond 1-day run → no events

    # Regression test for phase_offset_wrap24 (S1 Appendix Eq 49):
    # when sleep_midpoint_mle occurs shortly AFTER CBTmin_mle (small positive
    # difference), phase_offset must land in (-12, 12], NOT wrap all the way
    # around to near -24 as the old (buggy) formula produced.
    CBTmin_mle_test = 5.0
    sleep_midpoint_mle_test = 5.5
    phase_offset_test = CircadianPOMDP.phase_offset_wrap24(sleep_midpoint_mle_test, CBTmin_mle_test)
    @test isapprox(phase_offset_test, 0.5; atol=1e-9)
    @test !isapprox(phase_offset_test, -23.5; atol=1e-9)
    @test phase_offset_test > -12.0
    @test phase_offset_test <= 12.0
end

# Visual verification: save plots to test/output for inspection

@testset "plot_visual_verification" begin
    using Random
    using Dates
    using Plots
    include(joinpath(@__DIR__, "test_utils.jl"))

    Random.seed!(999)
    params = CircadianPOMDP.model_parameters()
    particles = CircadianPOMDP.initialize_particles(10, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=2,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    cbtmin_ev, onset_ev, offset_ev = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=1/60, verbose=false)

    out = test_output_dir()

    # plot_population_with_neuronal_dynamics (needs full histories; simulate separately)
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)
    plt = CircadianPOMDP.plot_population_with_neuronal_dynamics(PH)
    @test plt !== nothing
    save_path = joinpath(out, "population_neuronal_dynamics.png")
    Plots.savefig(plt, save_path)
    @test isfile(save_path)

    # plot_summary_histograms (needs MLE summaries)
    summaries = CircadianPOMDP.compute_mle_population_summaries(
        cbtmin_ev, onset_ev, offset_ev, particles; verbose=false, start_day=1.0, min_sleep_episodes=2)
    if !isempty(summaries)
        plt2 = CircadianPOMDP.plot_summary_histograms(summaries)
        @test plt2 !== nothing
        hist_path = joinpath(out, "summary_histograms.png")
        Plots.savefig(plt2, hist_path)
        @test isfile(hist_path)
    end
end

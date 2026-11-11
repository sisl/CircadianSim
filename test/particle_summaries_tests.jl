using Random
using Dates

@testset "particle_summaries" begin
    Random.seed!(111)
    params = CircadianPOMDP.model_parameters()
    particles = CircadianPOMDP.initialize_particles(5, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=3,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)

    # estimate_VonMeises
    phases = rand(100) * 2π .- π
    mu, Rbar, K = CircadianPOMDP.estimate_VonMeises(phases)
    @test mu isa Float64
    @test Rbar >= 0 && Rbar <= 1
    @test K >= 0

    # particle_summaries (legacy) — assert non-empty for a 3-day run
    t = collect(0:length(PH[1])-1) .* (1/60)
    sums = CircadianPOMDP.particle_summaries(PH, t)
    @test !isempty(sums)
    @test all(s -> hasproperty(s, :CBTmin_mle), sums)

    # particle_summaries: empty history does not crash (logs warning)
    empty_hist = [CircadianPOMDP.CircadianState[]]
    result = CircadianPOMDP.particle_summaries(empty_hist, [0.0])
    @test result isa Vector
    @test isempty(result)
end

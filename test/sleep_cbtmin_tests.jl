using Random
using Dates

@testset "sleep_CBTmin_detection" begin
    Random.seed!(321)
    params = CircadianPOMDP.model_parameters()
    particles = CircadianPOMDP.initialize_particles(5, DateTime(2023, 1, 1); state_params_mode=:sampled)
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=3,
        lux_on=10_000.0, light_on=7.0, light_off=23.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing
    )
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)

    # get_CBTmin returns vector of indices
    for i in eachindex(PH)
        idx = CircadianPOMDP.get_CBTmin(PH[i]; dt=1/60)
        @test idx isa Vector{Int}
        @test all(j -> 1 <= j <= length(PH[i]), idx)
    end

    # For a 3-day simulation, expect at least 2 CBTmin events (~1 per 24h)
    all_idx = CircadianPOMDP.get_CBTmin(PH[1]; dt=1/60)
    @test length(all_idx) >= 2

    # get_sleep_periods returns SleepPeriod vector
    for i in eachindex(PH)
        sleeps = CircadianPOMDP.get_sleep_periods(PH[i]; dt=1/60)
        @test sleeps isa Vector
        for sp in sleeps
            @test sp isa CircadianPOMDP.SleepPeriod
            @test sp.onset_idx isa Union{Int, Missing}
            @test sp.offset_idx isa Union{Int, Missing}
            @test sp.duration isa Union{Float64, Missing}
            if !ismissing(sp.onset_idx) && !ismissing(sp.offset_idx)
                @test sp.onset_idx < sp.offset_idx
            end
            if !ismissing(sp.duration)
                @test sp.duration > 0
            end
        end
    end

    # CBTmin and sleep periods from same history are consistent
    idx = CircadianPOMDP.get_CBTmin(PH[1]; dt=1/60)
    if !isempty(idx)
        @test idx[1] >= 1
    end
end

# CBTmin = time at which atan(xc, x) reaches −170.7° (St. Hilaire et al. 2007), plus 0.97 h. The phase decreases
# with time, so the threshold is reached from above; the −π → +π jump once per cycle must not count (it did, before
# the fix, which put CBTmin ~0.63 h late).
@testset "CBTmin phase convention" begin
    thr = CircadianPOMDP.CBTMIN_PHASE_THRESHOLD
    @test CircadianPOMDP.cbtmin_phase_crossing(thr + 0.01, thr - 0.01) ≈ 0.5
    @test CircadianPOMDP.cbtmin_phase_crossing(thr - 0.01, thr + 0.01) === nothing   # upward: not a crossing
    @test CircadianPOMDP.cbtmin_phase_crossing(-π + 1e-3, π - 1e-3) === nothing      # wrap: not a crossing
    @test CircadianPOMDP.cbtmin_phase_crossing(thr + 0.02, thr + 0.01) === nothing

    params = CircadianPOMDP.model_parameters()
    t0 = DateTime(2023, 1, 1)
    dt = 1/60
    particles = [CircadianPOMDP.default_circadian_state()]
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=5, lux_on=150.0, light_on=CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H,
        light_off=CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_OFF_H, lux_off=0.0, sleep_light=missing, sleep_dark=missing)
    PH = CircadianPOMDP.simulate_particle_from_initial_population(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep)
    states = PH[1]
    phase = atan.([s.xc for s in states], [s.x for s in states])
    @test count(<(0), diff(phase)) > 0.99 * (length(phase) - 1)                     # phase decreases
    shift = round(Int, CircadianPOMDP.CBTMIN_TEMPORAL_OFFSET_H / dt)
    idx = CircadianPOMDP.get_CBTmin(states; dt=dt)
    @test length(idx) >= 4
    for i in idx
        @test abs(rad2deg(phase[i - shift]) - (-170.7)) < 0.5                       # one step before the threshold
    end

    # The on-the-fly detector (used for the stored histories) agrees with get_CBTmin to within a step
    ev, _, _ = CircadianPOMDP.simulate_particle_population_events_only(
        particles, LD, params, CircadianPOMDP.circadian_transition_sleep, t0; dt=dt, verbose=false)
    live_h = sort([e.t_hours for e in ev])
    traj_h = (idx .- 1) .* dt
    n = min(length(live_h), length(traj_h))
    @test n >= 4
    @test maximum(abs.(live_h[end-n+1:end] .- traj_h[end-n+1:end])) <= 2dt
end

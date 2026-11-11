using StableRNGs

@testset "circadian_transition" begin
    params = CircadianPOMDP.model_parameters()
    s0 = CircadianPOMDP.default_circadian_state()
    a = CircadianPOMDP.default_circadian_action(0.0, 1/60)

    # transition() API (single entry point) — CircadianAction
    s1 = CircadianPOMDP.transition(s0, a, params)
    @test s1 isa CircadianPOMDP.CircadianState
    @test s1.x != s0.x || s1.xc != s0.xc
    @test s1.n >= 0 && s1.n <= 1
    @test isfinite(s1.x) && isfinite(s1.xc) && (s1.x^2 + s1.xc^2) > 0.1

    # transition with light
    a_light = CircadianPOMDP.CircadianAction(1000.0, missing, 1/60)
    s_light = CircadianPOMDP.transition(s0, a_light, params)
    @test s_light isa CircadianPOMDP.CircadianState

    # circadian_transition_sleep
    s_sleep = CircadianPOMDP.circadian_transition_sleep(s0, a, params)
    @test s_sleep isa CircadianPOMDP.CircadianState

    # Simulate short trajectory (transition_fn by name)
    LD = CircadianPOMDP.generate_light_diet_basic(days=1, lux_on=0.0, light_on=12.0, light_off=12.0,
        lux_off=0.0, sleep_light=missing, sleep_dark=missing)
    hist = CircadianPOMDP.simulate_particle_from_initial_population(
        [s0], LD, params, CircadianPOMDP.circadian_transition_sleep)
    @test length(hist) == 1
    @test length(hist[1]) == length(LD) + 1
    @test hist[1][1] == s0

    # simulate_particle_population default uses transition()
    hist2 = CircadianPOMDP.simulate_particle_population(2, LD, params;
        init_fn=N -> CircadianPOMDP.initialize_particles(N, s0.t; state_params_mode=:default))
    @test length(hist2) == 2
    @test all(h -> length(h) == length(LD) + 1, hist2)

    # transition with stochastic light (TruncNormal per Appendix A.4.1; default eyelid fraction during sleep)
    s_stoch = CircadianPOMDP.transition(s0, a_light, params; deterministic_light=false)
    @test s_stoch isa CircadianPOMDP.CircadianState
    @test isfinite(s_stoch.x) && isfinite(s_stoch.xc)
    @test 0.0 <= s_stoch.I_eye <= 10000.0

    # Stochastic illuminance: reproducible with fixed RNG; in support [0, 10^4]; mean near prescribed for many draws
    a1k = CircadianPOMDP.CircadianAction(1000.0, false, 1/60)
    r_a = StableRNG(777)
    r_b = StableRNG(777)
    s_a = CircadianPOMDP.transition(s0, a1k, params; deterministic_light=false, rng=r_a)
    s_b = CircadianPOMDP.transition(s0, a1k, params; deterministic_light=false, rng=r_b)
    @test s_a.I_eye == s_b.I_eye
    acc = 0.0
    for ℓ in 1:4000
        r = StableRNG(ℓ)
        s = CircadianPOMDP.transition(s0, a1k, params; deterministic_light=false, rng=r)
        @test 0.0 <= s.I_eye <= 10000.0
        acc += s.I_eye
    end
    @test isapprox(acc / 4000, 1000.0, rtol=0.06)
end

@testset "Process C: phi delays the circadian drive (rotation)" begin
    params = CircadianPOMDP.model_parameters()
    p_H = params.params_H
    s0 = CircadianPOMDP.default_circadian_state()
    a = CircadianPOMDP.CircadianAction(0.0, missing, 1e-9)   # tiny step: x_new ≈ x, xc_new ≈ xc
    with(s; kw...) = CircadianPOMDP.CircadianState((haskey(kw, f) ? kw[f] : getfield(s, f)
                                                     for f in fieldnames(CircadianPOMDP.CircadianState))...)
    # C is recovered from the stored threshold H_plus = (2.46 - A_v + v_vc C) / v_vh
    C_of(s) = (s.H_plus * p_H[:v_vh] - 2.46 + p_H[:A_v]) / p_H[:v_vc]
    C_at(α, phi) = C_of(CircadianPOMDP.transition(with(s0; x=sin(α), xc=cos(α), phi=phi), a, params))
    C_prior(x, xc) = 0.5 * (1.0 - 0.16 * xc + 0.8 * x)

    # phi = 0 is the unshifted drive
    s1 = CircadianPOMDP.transition(with(s0; phi=0.0), CircadianPOMDP.CircadianAction(0.0, missing, 1/60), params)
    @test C_of(s1) ≈ C_prior(s1.x, s1.xc) atol=1e-12

    # On the unit circle, phi hours of offset evaluates the unshifted drive 2π·phi/τ earlier in phase
    τ = s0.tau_x
    for phi in (-3.0, -1.0, 0.5, 2.0, 6.0), α in range(0, 2π; length=13)[1:end-1]
        @test C_at(α, phi) ≈ C_at(α - 2π * phi / τ, 0.0) atol=1e-7
    end

    # A rotation shifts the drive in time without changing its range; phi = τ is a full cycle
    αs = range(0, 2π; length=721)
    C0 = [C_at(α, 0.0) for α in αs]
    C3 = [C_at(α, 3.0) for α in αs]
    @test maximum(C3) ≈ maximum(C0) atol=1e-4
    @test minimum(C3) ≈ minimum(C0) atol=1e-4
    @test all(isapprox(C_at(α, τ), C_at(α, 0.0); atol=1e-7) for α in αs[1:60:end])
end

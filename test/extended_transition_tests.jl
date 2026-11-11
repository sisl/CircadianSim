using Dates

@testset "circadian_transition_ext" begin
    params = CircadianPOMDP.model_parameters()
    t0 = DateTime(2023, 1, 1, 0, 0, 0)
    s_ext = CircadianPOMDP.initialize_particles(1, t0; latitude=37.77, longitude=-122.42, timezone="America/Los_Angeles")[1]
    t1 = t0 + Hour(12)
    action = CircadianPOMDP.CircadianActionExt(
        "Regular_V4_061hEarlier", "open", missing,
        1/60, t0, t0 + Day(1)
    )

    # is_within_action_window
    @test CircadianPOMDP.is_within_action_window(s_ext, action) == true
    @test CircadianPOMDP.is_within_action_window(t0, action) == true
    @test CircadianPOMDP.is_within_action_window(t0 + Day(2), action) == false

    # get_action_duration_hours
    dur = CircadianPOMDP.get_action_duration_hours(action)
    @test dur ≈ 24.0

    # get_action_num_steps
    n_steps = CircadianPOMDP.get_action_num_steps(action)
    @test n_steps == 24 * 60  # 24h * 60 steps/hour

    # create_action_sequence
    a1 = CircadianPOMDP.CircadianActionExt("diet", "open", missing, 1/60, t0, t0 + Hour(12))
    a2 = CircadianPOMDP.CircadianActionExt("diet", "open", missing, 1/60, t0 + Hour(12), t0 + Day(1))
    seq = CircadianPOMDP.create_action_sequence([a2, a1])
    @test seq[1].startdate == t0
    @test seq[2].startdate == t0 + Hour(12)

    # get_current_action
    curr = CircadianPOMDP.get_current_action(t0 + Hour(6), [a1, a2])
    @test curr !== nothing
    @test curr.startdate == t0

    curr_none = CircadianPOMDP.get_current_action(t0 + Day(2), [a1, a2])
    @test curr_none === nothing

    # circadian_transition_ext (single step; deterministic for reproducible tests)
    s_next = CircadianPOMDP.circadian_transition_ext(s_ext, action, params; stochastic_light=false)
    @test s_next isa CircadianPOMDP.CircadianState
    @test s_next.t > s_ext.t
    # Amplitude may vary; ensure state is finite
    @test isfinite(s_next.x) && isfinite(s_next.xc)

    # transition() API with CircadianActionExt
    s_next2 = CircadianPOMDP.transition(s_ext, action, params; deterministic_light=true)
    @test s_next2 isa CircadianPOMDP.CircadianState
    @test s_next2.t > s_ext.t

    # Same Regular_V4 schedule, 150 lux diet name (force awake so ocular light reflects prescribed lux, not sleep mask)
    t_noon = t0 + Hour(12)
    s_noon = CircadianPOMDP.CircadianState(
        s_ext.n, s_ext.x, s_ext.xc, s_ext.H, s_ext.Vm, s_ext.Vv, false,
        s_ext.H_plus, s_ext.H_minus, s_ext.tau_x, s_ext.K, s_ext.chi, s_ext.mu_h, s_ext.phi,
        s_ext.I_eye, s_ext.latitude, s_ext.longitude, s_ext.timezone, t_noon)
    action_150 = CircadianPOMDP.CircadianActionExt(
        "Regular_V4_061hEarlier_150lux", "open", missing,
        1/60, t0, t0 + Day(1))
    s_hi_noon = CircadianPOMDP.transition(s_noon, action, params; deterministic_light=true)
    s_lo_noon = CircadianPOMDP.transition(s_noon, action_150, params; deterministic_light=true)
    @test s_hi_noon.I_eye > s_lo_noon.I_eye

    # Edge case: get_current_action with empty action list returns nothing
    empty_result = CircadianPOMDP.get_current_action(t0, CircadianPOMDP.CircadianActionExt[])
    @test empty_result === nothing

    # circadian_transition_ext_circadian_only (no sleep dynamics; not exported)
    s_circ_only = CircadianPOMDP.circadian_transition_ext_circadian_only(s_ext, action, params)
    @test s_circ_only isa CircadianPOMDP.CircadianState
    @test s_circ_only.t > s_ext.t
    @test isfinite(s_circ_only.x) && isfinite(s_circ_only.xc)
end

using Dates

@testset "light_diet" begin
    # generate_light_diet_basic
    LD = CircadianPOMDP.generate_light_diet_basic(
        days=2,
        lux_on=10_000.0,
        light_on=7.0,
        light_off=23.0,
        lux_off=0.0,
        sleep_light=missing,
        sleep_dark=missing
    )
    @test length(LD) == 2 * 24 * 60
    @test all(a -> a isa CircadianPOMDP.CircadianAction, LD)
    @test all(a -> a.dt ≈ 1/60, LD)
    @test LD[1].I_act ≈ 0.0
    @test LD[7*60 + 1].I_act ≈ 10_000.0
    @test LD[23*60].I_act ≈ 10_000.0
    @test LD[23*60 + 1].I_act ≈ 0.0

    # generate_shifted_dark_diet
    dark_diet = CircadianPOMDP.generate_shifted_dark_diet(1, 0.0, 8.0, 10000.0; shift=0.0)
    @test length(dark_diet) == 24 * 60

    # generate_light_period_diet
    light_diet = CircadianPOMDP.generate_light_period_diet(1, 8.0, 24.0, 10000.0; shift=0.0)
    @test length(light_diet) == 24 * 60

    # generate_intervention_light_diet: forced wake only during prescribed bright minutes; dark = missing
    interv = CircadianPOMDP.generate_intervention_light_diet(8.0, 2.0, 10000.0; days=1, shift=0.0)
    @test length(interv) == 24 * 60
    for a in interv
        if a.I_act > 0
            @test a.sleep_forced === false
        else
            @test a.sleep_forced === missing
        end
    end
    # Wrapped-window day (shift=0): bright outside [dark_start, dark_end)
    interv_wrapped = CircadianPOMDP.generate_intervention_light_diet(22.0, 4.0, 10000.0; days=1, shift=0.0)
    for a in interv_wrapped
        if a.I_act > 0
            @test a.sleep_forced === false
        else
            @test a.sleep_forced === missing
        end
    end
    # Same rule for a dimmer prescribed intervention lux (still forced wake while light is on)
    interv_dim = CircadianPOMDP.generate_intervention_light_diet(8.0, 2.0, 150.0; days=1, shift=0.0)
    for a in interv_dim
        if a.I_act > 0
            @test a.sleep_forced === false
        else
            @test a.sleep_forced === missing
        end
    end
    # Target-segment style: still open sleep everywhere
    tgt = CircadianPOMDP.generate_light_period_diet(1, 8.0, 24.0, 150.0; shift=0.0, force_wake_during_lux=false)
    @test all(a -> a.sleep_forced === missing, tgt)

    # get_intervention_policies
    policies = CircadianPOMDP.get_intervention_policies()
    @test length(policies) == 300
    @test all(p -> 0 <= p.start <= 23 && 1 <= p.duration <= 24 - p.start, policies)

    # generate_light_diets_for_condition (PST + targets: no forced sleep/wake anywhere)
    condition = (baseline_lux=10000.0, target_lux=10000.0, intervention_lux=10000.0)
    LD_PST, targets = CircadianPOMDP.generate_light_diets_for_condition(condition; baseline_days=2, target_days=3)
    @test length(LD_PST) == 2 * 24 * 60
    @test length(targets) == 23
    @test length(targets[1]) == 3 * 24 * 60
    for a in LD_PST
        @test a.sleep_forced === missing
    end
    for t in targets
        @test all(x -> x.sleep_forced === missing, t)
    end

    # build_baseline_diet_full
    full_baseline = CircadianPOMDP.build_baseline_diet_full(condition, 1; baseline_days=2, target_days=3)
    @test length(full_baseline) == length(LD_PST) + length(targets[1])

    # build_intervention_diet_full: PST + target = all missing; intervention slice = false when I_act > 0
    policy = (start=8.0, duration=2.0)
    full_interv = CircadianPOMDP.build_intervention_diet_full(condition, policy, 1; baseline_days=2, target_days=3)
    @test length(full_interv) == length(LD_PST) + 24*60 + length(targets[1])
    n_pst = length(LD_PST)
    n_1d = 24 * 60
    @test all(a -> a.sleep_forced === missing, full_interv[1:n_pst])
    @test all(a -> a.sleep_forced === missing, full_interv[n_pst + n_1d + 1:end])
    for a in full_interv[n_pst + 1:n_pst + n_1d]
        if a.I_act > 0
            @test a.sleep_forced === false
        else
            @test a.sleep_forced === missing
        end
    end

    # get_available_light_diets (JetLag: Regular_V4 10k + 150 lux)
    diets = CircadianPOMDP.get_available_light_diets()
    @test diets isa AbstractVector
    @test "Regular_V4_061hEarlier" in diets
    @test "Regular_V4_061hEarlier_150lux" in diets

    # get_light_intensity (Regular_V4 is clock-based)
    t = DateTime(2023, 6, 21, 12, 0, 0)
    I = CircadianPOMDP.get_light_intensity("Regular_V4_061hEarlier", t, 37.77, "lux_20260117"; lon_offset=0.0)
    @test I isa Tuple
    @test length(I) >= 1
    @test I[1] >= 0
    @test I[2] === missing  # regular on-the-fly schedule: no forced sleep/wake from diet

    # Nominal dark midnight–8:00, light 8:00–24:00; NHANES wake shift applied (same clock for 10k / 150 lux diet)
    @test CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H ≈ 8.0 - CircadianPOMDP.REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H
    t_dark = DateTime(2023, 6, 21, 6, 0, 0)   # in dark (before ON h)
    t_on = DateTime(2023, 6, 21, 10, 0, 0)   # in light (16 h window)
    @test CircadianPOMDP.get_light_intensity("Regular_V4_061hEarlier", t_dark, 37.77, "lux_20260117")[1] == 0.0
    @test CircadianPOMDP.get_light_intensity("Regular_V4_061hEarlier", t_on, 37.77, "lux_20260117")[1] == 10000.0
    @test CircadianPOMDP.get_light_intensity("Regular_V4_061hEarlier_150lux", t_on, 37.77, "lux_20260117")[1] == 150.0

    # get_timezone_center_lon
    lon = CircadianPOMDP.get_timezone_center_lon("America/Los_Angeles")
    @test lon isa Float64
    @test -180 <= lon <= 180

    # infer_timezone_center_from_longitude
    tz_lon = CircadianPOMDP.infer_timezone_center_from_longitude(-122.0)
    @test tz_lon isa Float64

    # JETLAG_MEAN_ALIGNMENT_SHIFT_H matches regular-schedule NHANES photoperiod shift
    @test CircadianPOMDP.JETLAG_MEAN_ALIGNMENT_SHIFT_H == CircadianPOMDP.REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H

    # get_light_and_sleep
    I2, sleep2 = CircadianPOMDP.get_light_and_sleep(t, 37.77, -122.42, "Regular_V4_061hEarlier", "lux_20260117")
    @test I2 >= 0
    @test sleep2 isa Union{Bool, Missing}

    # compute_light_from_action
    I3, sleep3 = CircadianPOMDP.compute_light_from_action(t, 37.77, -122.42, "Regular_V4_061hEarlier", "lux_20260117")
    @test I3 >= 0
end

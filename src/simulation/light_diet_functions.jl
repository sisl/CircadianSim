# ================================================================
# Jet Lag Light Diet Builders
# ================================================================
# Array-based light diet generators for the JetLag experiment pipeline:
#   - Shifted dark/light period generators (for timezone shifts)
#   - Intervention diet (single-day light window)
#   - Policy enumeration (all 300 (start, duration) pairs)
#   - Full scenario constructors (baseline + target, with/without intervention)
#
# Sleep policy: `sleep_forced` is `missing` (open sleep/wake) on the **regular PST**
# segment and on **post-shift target** segments. The **one-day intervention** uses `missing`
# during dark (0 lux) and forced wake (`sleep_forced = false`) during prescribed bright minutes.
#
# Depends on: generate_light_diet_basic (generate_light_diet_basic.jl)
#             light_intensity_core (light_intensity_core.jl)

"""Shift (hours) for array-based jet-lag diets (timezone segments, intervention day). Same as
`REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H` (150 lux NHANES photoperiod calibration, 1.341 h) so jet-lag
schedules match on-the-fly `Regular_V4_061hEarlier*` clock boundaries."""
const JETLAG_MEAN_ALIGNMENT_SHIFT_H = REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H

"""
Generate a light-diet segment for a **dark period** [dark_start_h, dark_end_h], shifted by `shift` hours.
"""
function generate_shifted_dark_diet(days::Int, dark_start_h::Float64, dark_end_h::Float64,
                                    lux_off::Float64; shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H)
    dark_start_shifted = mod(dark_start_h - shift + 24.0, 24.0)
    dark_end_shifted   = mod(dark_end_h   - shift + 24.0, 24.0)
    if dark_start_shifted > dark_end_shifted
        light_start = dark_end_shifted
        light_end   = dark_start_shifted
        return generate_light_diet_basic(
            days=days, lux_on=lux_off, light_on=light_start, light_off=light_end,
            lux_off=0.0, sleep_light=missing, sleep_dark=missing
        )
    else
        return generate_light_diet_basic(
            days=days, lux_on=0.0, light_on=dark_start_shifted, light_off=dark_end_shifted,
            lux_off=lux_off, sleep_light=missing, sleep_dark=missing
        )
    end
end

"""
For `generate_light_diet_basic`, `[light_on, light_off)` carries `lux_on`; the rest carries `lux_off`.
Simple day: that interval is **bright**. Wrapped bright window (crosses midnight): that interval is the **dark** core.
When forcing wake only during bright `lux`, `(sleep_light, sleep_dark)` must swap between those layouts.
"""
function _sleep_for_basic_interval(force_wake::Bool, interval_is_dark_core::Bool)
    if !force_wake
        return (missing, missing)
    end
    if interval_is_dark_core
        return (missing, false)   # inside interval = dark → missing; outside = bright → forced wake
    end
    return (false, missing)     # inside = bright → forced wake; outside = dark → missing
end

"""
Generate a light-diet segment for a **light period** [light_start_h, light_end_h], shifted by `shift` hours.

If `force_wake_during_lux` is true, minutes that receive `lux` use `sleep_forced = false` (forced awake);
minutes at 0 lux use `missing` (unconstrained sleep). Jet-lag **target** segments keep the default `false`;
the **one-day intervention** sets this to `true` to match bright-light therapy while allowing sleep in the dark.
"""
function generate_light_period_diet(days::Int, light_start_h::Float64, light_end_h::Float64,
                                    lux::Float64; shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H,
                                    force_wake_during_lux::Bool=false)
    light_start_shifted = mod(light_start_h - shift + 24.0, 24.0)
    light_end_shifted   = mod(light_end_h   - shift + 24.0, 24.0)
    wrapped = light_start_shifted > light_end_shifted
    sleep_light, sleep_dark = _sleep_for_basic_interval(force_wake_during_lux, wrapped)
    if wrapped
        return generate_light_diet_basic(
            days=days, lux_on=0.0, light_on=light_end_shifted, light_off=light_start_shifted,
            lux_off=lux, sleep_light=sleep_light, sleep_dark=sleep_dark
        )
    end
    return generate_light_diet_basic(
        days=days, lux_on=lux, light_on=light_start_shifted, light_off=light_end_shifted,
        lux_off=0.0, sleep_light=sleep_light, sleep_dark=sleep_dark
    )
end

"""
Generate **intervention** light diet: one day with a single light window at `intervention_lux`.
`light_start` and `light_duration` are in hours (clock hours, before the NHANES shift).

Forced wake (`sleep_forced = false`) during bright minutes; `missing` during 0 lux (experiments use
10k via `light_conditions_specs()` in `experiments/jetlag_reference.jl`).
"""
function generate_intervention_light_diet(light_start::Float64, light_duration::Float64,
                                          intervention_lux::Float64; days::Int=1,
                                          shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H, verbose::Bool=false)
    light_end_h = light_start + light_duration
    ld = generate_light_period_diet(days, light_start, light_end_h, intervention_lux; shift=shift,
                                    force_wake_during_lux=true)
    if verbose
        ls = mod(light_start - shift + 24.0, 24.0)
        le = mod(light_end_h  - shift + 24.0, 24.0)
        println("   Intervention: Light $(round(ls, digits=2))h–$(round(le, digits=2))h, $(light_duration)h at $(intervention_lux) lux")
    end
    return ld
end

"""
Return all 300 intervention policies: `(start, duration)` with `start` ∈ 0:23, `duration` ∈ 1:(24-start).
"""
function get_intervention_policies()
    return [(start=Float64(s), duration=Float64(d)) for s in 0:23 for d in 1:(24 - s)]
end

"""
Build baseline PST diet and 23 target-timezone diets for `condition`.
"""
function generate_light_diets_for_condition(condition;
                                            baseline_light_on::Float64=REGULAR_SCHEDULE_LIGHT_ON_H,
                                            baseline_light_off::Float64=REGULAR_SCHEDULE_LIGHT_OFF_H,
                                            baseline_days::Int=35,
                                            target_days::Int=50,
                                            shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H)
    LD_PST = generate_light_diet_basic(
        days=baseline_days,
        lux_on=condition.baseline_lux,
        light_on=baseline_light_on,
        light_off=baseline_light_off,
        lux_off=0.0,
        sleep_light=missing,
        sleep_dark=missing
    )
    target_timezones = []
    for tz in 1:16
        push!(target_timezones, generate_shifted_dark_diet(target_days, Float64(tz), Float64(tz + 8), condition.target_lux; shift=shift))
    end
    for tz in 17:23
        push!(target_timezones, generate_light_period_diet(
            target_days, Float64(tz - 16), Float64(tz), condition.target_lux; shift=shift,
            force_wake_during_lux=false))  # open sleep/wake only; intervention day is separate
    end
    return LD_PST, target_timezones
end

"""
Full baseline diet: `baseline_days` PST + `target_days` target timezone (no intervention).
"""
function build_baseline_diet_full(condition, tz_index::Int;
                                  baseline_light_on::Float64=REGULAR_SCHEDULE_LIGHT_ON_H,
                                  baseline_light_off::Float64=REGULAR_SCHEDULE_LIGHT_OFF_H,
                                  baseline_days::Int=35,
                                  target_days::Int=50,
                                  shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H)
    LD_PST, target_timezones = generate_light_diets_for_condition(condition;
        baseline_light_on=baseline_light_on, baseline_light_off=baseline_light_off,
        baseline_days=baseline_days, target_days=target_days, shift=shift)
    return vcat(LD_PST, target_timezones[tz_index])
end

"""
Full intervention diet: `baseline_days` PST + 1-day intervention + `target_days` target timezone.
"""
function build_intervention_diet_full(condition, policy, tz_index::Int;
                                      baseline_light_on::Float64=REGULAR_SCHEDULE_LIGHT_ON_H,
                                      baseline_light_off::Float64=REGULAR_SCHEDULE_LIGHT_OFF_H,
                                      baseline_days::Int=35,
                                      target_days::Int=50,
                                      shift::Float64=JETLAG_MEAN_ALIGNMENT_SHIFT_H)
    LD_PST, target_timezones = generate_light_diets_for_condition(condition;
        baseline_light_on=baseline_light_on, baseline_light_off=baseline_light_off,
        baseline_days=baseline_days, target_days=target_days, shift=shift)
    intervention_diet = generate_intervention_light_diet(
        policy.start, policy.duration, condition.intervention_lux; days=1, shift=shift
    )
    return vcat(LD_PST, intervention_diet, target_timezones[tz_index])
end

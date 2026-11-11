# ================================================================
# Light Intensity Core
# ================================================================
# General-purpose light intensity infrastructure:
#   - Timezone/longitude helpers (used by circadian_transition_ext)
#   - On-the-fly light diet functions for CircadianActionExt
#   - LIGHT_DIETS registry and dispatcher
#
# Jet-lag-specific diet builders are in light_diet_functions.jl.

using Dates

# ================================================================
# HELPERS
# ================================================================

"""
    get_hour_float(t::DateTime) -> Float64

Hour of day as a float (e.g. 9:30 AM = 9.5).
"""
function get_hour_float(t::DateTime)
    return Dates.hour(t) + Dates.minute(t) / 60.0 +
           Dates.second(t) / 3600.0 + Dates.millisecond(t) / 3600000.0
end

"""
    get_timezone_center_lon(timezone::String; longitude::Union{Float64, Nothing}=nothing) -> Float64

Center longitude for a timezone string. Used by circadian_transition_ext for lon_offset.
Defaults to -120.0 (Pacific) if not recognized.
"""
function get_timezone_center_lon(timezone::String; longitude::Union{Float64, Nothing}=nothing)
    if occursin("Eastern", timezone) || occursin("New_York", timezone)
        return -75.0
    elseif occursin("Central", timezone) || occursin("Chicago", timezone)
        return -90.0
    elseif occursin("Mountain", timezone) || occursin("Denver", timezone)
        return -105.0
    elseif occursin("Pacific", timezone) || occursin("Los_Angeles", timezone)
        return -120.0
    elseif occursin("Alaska", timezone) || occursin("Anchorage", timezone)
        return -135.0
    elseif occursin("Hawaii", timezone) || occursin("Honolulu", timezone) || occursin("Aleutian", timezone)
        return -150.0
    elseif occursin("Samoa", timezone)
        return -165.0
    elseif occursin("Chamorro", timezone)
        return 150.0
    elseif occursin("Atlantic", timezone) || occursin("Puerto_Rico", timezone)
        return -60.0
    elseif timezone == "Unknown" && longitude !== nothing
        return infer_timezone_center_from_longitude(longitude)
    else
        return -120.0
    end
end

"""
    infer_timezone_center_from_longitude(longitude::Float64) -> Float64

Infer timezone center longitude from longitude (fallback when timezone is "Unknown").
"""
function infer_timezone_center_from_longitude(longitude::Float64)
    lon = longitude
    if lon > 100.0
        return 150.0
    elseif lon > 0.0
        @warn "Unexpected positive longitude $lon, defaulting to Pacific"
        return -120.0
    end
    if lon > -67.5
        return -60.0
    elseif lon > -82.5
        return -75.0
    elseif lon > -97.5
        return -90.0
    elseif lon > -112.5
        return -105.0
    elseif lon > -127.5
        return -120.0
    elseif lon > -142.5
        return -135.0
    elseif lon > -157.5
        return -150.0
    else
        return -165.0
    end
end

# ================================================================
# ON-THE-FLY LIGHT DIET FUNCTIONS (used by CircadianActionExt)
# ================================================================
# 16 h light / 8 h dark: dark midnight–8:00, light 8:00–24:00 (nominal). Photoperiod shift below is the **150 lux**
# NHANES wake-alignment calibration: mean simulated sleep offset at 150 lux (revision baseline, included particles)
# equals the mean NHANES 2021–2023 weekday wake time (experiments/2b_SensitivityRevision/calibrate_nhanes_alignment.jl;
# was 1.143 h before the Vm/Vv fix and revision baseline, 1.248 h before Process C became a rotation by ψ, with ψ fixed at 0). The same `REGULAR_SCHEDULE_LIGHT_ON_H` /
# `REGULAR_SCHEDULE_LIGHT_OFF_H` apply to both 10k and 150 lux on-the-fly diets (`Regular_V4_061hEarlier*`).
# Array-based jet-lag diets use the same shift via `JETLAG_MEAN_ALIGNMENT_SHIFT_H` in light_diet_functions.jl.
const REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H = 1.341
const REGULAR_SCHEDULE_LIGHT_ON_H  = 8.0 - REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H
const REGULAR_SCHEDULE_LIGHT_OFF_H = 24.0 - REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H

"""
    light_diet_Regular_V4_061hEarlier(t, latitude, solar_model; lon_offset=0.0, eye_fraction=0.10)

Regular (baseline) schedule: 8 h dark (midnight–8:00 nominal), 16 h light (8:00–midnight nominal); boundaries are
`REGULAR_SCHEDULE_LIGHT_ON_H`–`REGULAR_SCHEDULE_LIGHT_OFF_H` after NHANES wake alignment. Returns 10k lux when on;
use diet name `Regular_V4_061hEarlier_150lux` for 150 lux (same schedule).
Open sleep.
"""
function light_diet_Regular_V4_061hEarlier(t::DateTime, latitude::Float64, solar_model::String;
                                            lon_offset::Float64=0.0, eye_fraction::Float64=0.10)
    hour_f = get_hour_float(t)
    I_act = (hour_f >= REGULAR_SCHEDULE_LIGHT_ON_H && hour_f < REGULAR_SCHEDULE_LIGHT_OFF_H) ? 10000.0 : 0.0
    return (I_act, missing)
end

function light_diet_Regular_V4_061hEarlier_150lux(t::DateTime, latitude::Float64, solar_model::String;
                                                 lon_offset::Float64=0.0, eye_fraction::Float64=0.10)
    hour_f = get_hour_float(t)
    I_act = (hour_f >= REGULAR_SCHEDULE_LIGHT_ON_H && hour_f < REGULAR_SCHEDULE_LIGHT_OFF_H) ? 150.0 : 0.0
    return (I_act, missing)
end

# ================================================================
# LIGHT DIET REGISTRY
# ================================================================

const LIGHT_DIETS = Dict{String, Function}(
    "Regular_V4_061hEarlier"        => light_diet_Regular_V4_061hEarlier,
    "Regular_V4_061hEarlier_150lux" => light_diet_Regular_V4_061hEarlier_150lux,
)

"""
    get_available_light_diets() -> Vector{String}

Return all registered light diet names.
"""
function get_available_light_diets()
    return sort(collect(keys(LIGHT_DIETS)))
end

"""
    get_light_intensity(light_diet, t, latitude, solar_model; lon_offset=0.0, eye_fraction=0.10)

Get `(I_act, sleep_forced)` for the named light diet at time `t` and location.
`solar_model` is ignored for the built-in `Regular_V4` diets.
"""
function get_light_intensity(light_diet::String, t::DateTime, latitude::Float64, solar_model::Union{String, Missing};
                             lon_offset::Float64=0.0, eye_fraction::Float64=0.10)
    if !haskey(LIGHT_DIETS, light_diet)
        error("Unknown light diet: $light_diet. Available: $(get_available_light_diets())")
    end
    solar_model_used = coalesce(solar_model, "lux_20260117")
    fn = LIGHT_DIETS[light_diet]
    return fn(t, latitude, solar_model_used; lon_offset=lon_offset, eye_fraction=eye_fraction)
end

"""
    get_light_and_sleep(t, latitude, longitude, light_diet, solar_model; timezone_center_lon=-120.0, eye_fraction=0.10)

Compute `(I_act, sleep_forced)` with `lon_offset = longitude - timezone_center_lon`.
"""
function get_light_and_sleep(t::DateTime, latitude::Float64, longitude::Float64,
                             light_diet::String, solar_model::Union{String, Missing};
                             timezone_center_lon::Float64=-120.0, eye_fraction::Float64=0.10)
    lon_offset = longitude - timezone_center_lon
    return get_light_intensity(light_diet, t, latitude, solar_model; lon_offset=lon_offset, eye_fraction=eye_fraction)
end

"""
    compute_light_from_action(state_t, state_lat, state_lon, action_light_diet, action_solar_model; ...)

Entry point for `CircadianState` + `CircadianActionExt` in `circadian_transition`.
"""
function compute_light_from_action(state_t::DateTime, state_lat::Float64, state_lon::Float64,
                                   action_light_diet::String, action_solar_model::Union{String, Missing};
                                   timezone_center_lon::Float64=-120.0, eye_fraction::Float64=0.10)
    return get_light_and_sleep(state_t, state_lat, state_lon,
                               action_light_diet, action_solar_model;
                               timezone_center_lon=timezone_center_lon, eye_fraction=eye_fraction)
end

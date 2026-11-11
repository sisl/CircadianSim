# State Transition functions
# Based on the St. Hilaire model (2007).
#
# PUBLIC API:
#   transition(state, action, params; kwargs...)  — single entry point; dispatches on action type.
#
# ACTION TYPES (AbstractCircadianAction):
#   CircadianAction (I_act, sleep_forced, dt): precomputed light.
#   CircadianActionExt (light_diet, ...): on-the-fly light from diet name and state time/location.
# All variants share one implementation and validated inputs.

# ================================================================
# Validation
# ================================================================

const _I_EYE_CLAMP = (0.0, 10000.0)

"""
Fraction of prescribed illuminance reaching the circadian photoreceptors during sleep (eyelid transmission).
Hong et al. use c ≈ 0.03.
"""
const EYELID_LIGHT_FRACTION = 0.03

function _validate_params(params)
    if !hasproperty(params, :params_C) || !hasproperty(params, :params_H)
        error("params must have params_C and params_H (e.g. from model_parameters())")
    end
    p_C, p_H = params.params_C, params.params_H
    for k in [:I_o, :beta, :G, :mu_c, :q, :p]
        haskey(p_C, k) || error("params.params_C missing required key :$k")
        isfinite(p_C[k]) || error("params.params_C.:$k must be finite")
    end
    for k in [:chi, :v_vh, :sigma, :tau_m, :tau_v, :Q_max, :theta_h]
        haskey(p_H, k) || error("params.params_H missing required key :$k")
        v = p_H[k]
        isfinite(v) || error("params.params_H.:$k must be finite")
        (k in (:chi, :v_vh, :sigma, :tau_m, :tau_v)) && (v <= 0) && error("params.params_H.:$k must be positive, got $v")
    end
    (p_C[:I_o] > 0 && p_C[:beta] >= 0) || error("params.params_C: I_o must be positive, beta non-negative")
    return nothing
end

function _validate_dt(dt::Real)
    if !isfinite(dt) || dt <= 0
        error("dt must be finite and positive, got dt = $dt")
    end
    return nothing
end

function _validate_transition_options(; sleep_light_fraction::Float64=0.0)
    if !(0 <= sleep_light_fraction <= 1)
        error("sleep_light_fraction must be in [0, 1], got $sleep_light_fraction")
    end
    return nothing
end

function _validate_sleep_forced(sleep_forced)
    if sleep_forced !== true && sleep_forced !== false && !ismissing(sleep_forced)
        error("sleep_forced must be true, false, or missing, got $(repr(sleep_forced))")
    end
    return nothing
end

"""Sanitize light intensity: replace NaN/Inf/negative with 0 and clamp to valid range."""
function _sanitize_I_act(I_act::Real)
    x = Float64(I_act)
    if !isfinite(x) || x < 0
        x = 0.0
    end
    return clamp(x, _I_EYE_CLAMP[1], _I_EYE_CLAMP[2])
end

"""Relative SD σ/μ for environmental illuminance noise (manuscript Appendix A.4.1: `TruncNormal(μ, 0.1 μ, 0, 10⁴)`)."""
const OCULAR_LIGHT_NOISE_REL_SD = 0.1

"""
Sample realized ocular illuminance when stochastic light is on: `TruncNormal(I_mean, OCULAR_LIGHT_NOISE_REL_SD * I_mean, 0, 10⁴)` lux.
For `I_mean ≤ 0`, returns `0.0` without sampling.
"""
function _sample_truncated_normal_illuminance(I_mean::Float64, rng::AbstractRNG)::Float64
    μ = I_mean
    if !(μ > 0.0)
        return 0.0
    end
    σ = OCULAR_LIGHT_NOISE_REL_SD * μ
    d = truncated(Normal(μ, σ), _I_EYE_CLAMP[1], _I_EYE_CLAMP[2])
    return clamp(rand(rng, d), _I_EYE_CLAMP[1], _I_EYE_CLAMP[2])
end

# ================================================================
# Internal implementation (single place for all dynamics)
# ================================================================

"""
    _circadian_transition_impl(state, I_act, sleep_forced, dt, params;
                              deterministic_light=true,
                              sleep_light_fraction=0.0,
                              rng=Random.default_rng()) -> CircadianState

Internal implementation used by all circadian transition variants.
Callers must pass validated params, dt, and options.
Pass an explicit `rng` for deterministic threaded simulation.
"""
function _circadian_transition_impl(
    state::CircadianState,
    I_act::Float64,
    sleep_forced::Union{Bool, Missing},
    dt::Float64,
    params;
    deterministic_light::Bool = true,
    sleep_light_fraction::Float64 = 0.0,
    rng::AbstractRNG = Random.default_rng(),
)
    _validate_params(params)
    _validate_dt(dt)
    _validate_transition_options(; sleep_light_fraction)
    _validate_sleep_forced(sleep_forced)
    I_act = _sanitize_I_act(I_act)

    p_C = params.params_C
    p_H = params.params_H

    I_o   = p_C[:I_o]
    beta  = p_C[:beta]
    G     = p_C[:G]
    mu_c  = p_C[:mu_c]
    q     = p_C[:q]
    p     = p_C[:p]

    n     = state.n
    x     = state.x
    xc    = state.xc
    H     = state.H
    Vm    = state.Vm
    Vv    = state.Vv
    sleep = state.sleep
    tau_x = state.tau_x
    K     = state.K
    chi   = state.chi
    mu_h  = state.mu_h
    phi   = state.phi

    # Resolve effective light at eyes
    is_sleeping = (sleep_forced === true) || (sleep_forced === missing && sleep)
    if is_sleeping
        I_raw = I_act * sleep_light_fraction
    else
        I_raw = I_act
    end
    if !deterministic_light
        I_raw = _sample_truncated_normal_illuminance(Float64(I_raw), rng)
    end
    # Clamp light to [0, 10000] lux: 10000 lux is the physiological upper bound for
    # indoor bright-light therapy; values above this are saturating and can cause
    # numerical issues in the Process L photoreceptor equations.
    I_eye = clamp(max(0.0, I_raw), 0.0, 10000.0)

    # Process L: α(t) = α_o (I_eye/I_o)^p · I_eye/(I_eye+100) with α_o = 0.1 (manuscript Table A1).
    # The factor 60 in dn links this α to minute-step discrete-time integration.
    if I_eye <= 0.0
        alpha = 0.0
    else
        alpha = 0.1 * (I_eye / I_o)^p * (I_eye / (I_eye + 100.0))
    end
    dn = 60.0 * (alpha * (1.0 - n) - beta * n)
    n_new = clamp(n + dt * dn, 0.0, 1.0)

    B_hat = alpha * G * (1.0 - n_new)
    B = B_hat * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)

    # Process P
    dx  = (π / 12.0) * (xc + mu_c * (x/3.0 + (4.0/3.0)*x^3 - (256.0/105.0)*x^7) + B)
    dxc = (π / 12.0) * (q*B*xc - x * ((24.0 / (0.99729 * tau_x))^2 + K*B))

    x_new  = x + dt * dx
    xc_new = xc + dt * dxc

    t_new = state.t + Dates.Millisecond(round(Int, dt * 3600.0 * 1000.0))

    # Process C: Circadian sleep drive, delayed by phi hours relative to the oscillator.
    # (x, xc) advances as (sin α, cos α), so rotating by θ = 2π·phi/τ evaluates the drive at α − θ;
    # phi = 0 is the unshifted drive 0.5(1 - 0.16 xc + 0.8 x).
    θ_phi  = 2.0 * π * phi / tau_x
    x_rot  = x_new * cos(θ_phi) - xc_new * sin(θ_phi)
    xc_rot = xc_new * cos(θ_phi) + x_new * sin(θ_phi)
    C = 0.5 * (1.0 - 0.16 * xc_rot + 0.8 * x_rot)

    v_mv    = p_H[:v_mv]
    theta_h = p_H[:theta_h]
    sigma   = p_H[:sigma]
    A_m     = p_H[:A_m]
    tau_m   = p_H[:tau_m]
    v_vm    = p_H[:v_vm]
    v_vc    = p_H[:v_vc]
    v_vh    = p_H[:v_vh]
    A_v     = p_H[:A_v]
    tau_v   = p_H[:tau_v]
    Q_max   = p_H[:Q_max]

    H_plus  = (2.46 - A_v + v_vc*C) / v_vh
    H_minus = (1.45 - A_v + v_vc*C) / v_vh

    Qm = Q_max / (1.0 + exp(-(Vm - theta_h) / sigma))
    Qv = Q_max / (1.0 + exp(-(Vv - theta_h) / sigma))

    if sleep_forced === true
        sleep_new = true
        dH = -H / chi
    elseif sleep_forced === false
        sleep_new = false
        dH = (-H + mu_h * Qm) / chi
    elseif sleep_forced === missing
        if H >= H_plus
            sleep_new = true
        elseif H <= H_minus
            sleep_new = false
        else
            sleep_new = sleep
        end
        dH = sleep_new ? (-H / chi) : ((-H + mu_h * Qm) / chi)
    else
        error("Invalid sleep_forced value: $sleep_forced")
    end

    H_new = H + dt * dH

    Vm_ss = -v_mv * Qv + A_m
    Vv_ss = -v_vm * Qm - v_vc * C + v_vh * H_new + A_v
    Vm_new = Vm_ss + (Vm - Vm_ss) * exp(-dt / tau_m)
    Vv_new = Vv_ss + (Vv - Vv_ss) * exp(-dt / tau_v)

    return CircadianState(
        n_new, x_new, xc_new, H_new, Vm_new, Vv_new,
        sleep_new, H_plus, H_minus,
        tau_x, K, chi, mu_h, phi,
        I_eye,
        state.latitude, state.longitude, state.timezone,
        t_new
    )
end

# ================================================================
# Unified public API
# ================================================================

"""
    transition(state, action, params; kwargs...) -> CircadianState

Main entry point for circadian dynamics. Dispatches on action type (CircadianAction or CircadianActionExt).

# Shared options
- `deterministic_light::Bool=true`: if `false`, realized illuminance follows Appendix A.4.1 (truncated normal).
- `sleep_light_fraction::Float64`: fraction of prescribed `I_act` used when the sleeper is **asleep** (eyelid transmission).
  Defaults to [`EYELID_LIGHT_FRACTION`](@ref) (~0.03) for **both** action types. Use `1.0` for no attenuation during sleep,
  or `0.0` to block light completely during sleep.

`CircadianActionExt` additionally accepts `light_diet_fn`, `timezone_center_lon`, `eye_fraction`, `check_date_window`.
"""
function transition(state::CircadianState, action::CircadianAction, params; rng::AbstractRNG=Random.default_rng(),
                    deterministic_light::Bool=true, sleep_light_fraction::Float64=EYELID_LIGHT_FRACTION, kwargs...)
    _validate_params(params)
    _validate_dt(action.dt)
    _validate_sleep_forced(action.sleep_forced)
    I_act = _sanitize_I_act(something(action.I_act, 0.0))
    return _circadian_transition_impl(
        state, I_act, action.sleep_forced, action.dt, params;
        deterministic_light=deterministic_light,
        sleep_light_fraction=sleep_light_fraction,
        rng=rng,
    )
end

function transition(state::CircadianState, action::CircadianActionExt, params; rng::AbstractRNG=Random.default_rng(),
                    deterministic_light::Bool=true, sleep_light_fraction::Float64=EYELID_LIGHT_FRACTION,
                    light_diet_fn::Function=get_light_intensity, timezone_center_lon::Float64=-120.0,
                    eye_fraction::Float64=0.10, check_date_window::Bool=false, kwargs...)
    _validate_params(params)
    _validate_dt(action.dt)
    if action.startdate >= action.stopdate
        error("CircadianActionExt requires startdate < stopdate, got [$(action.startdate), $(action.stopdate))")
    end

    t = state.t
    if check_date_window && (t < action.startdate || t >= action.stopdate)
        @warn "State time $(t) is outside action window [$(action.startdate), $(action.stopdate))"
    end

    timezone = state.timezone
    longitude = state.longitude
    latitude = state.latitude
    tz_center = (timezone != "" && timezone != "Unknown") ?
        get_timezone_center_lon(timezone) :
        get_timezone_center_lon("Unknown"; longitude=longitude)
    lon_offset = longitude - tz_center

    solar_model = coalesce(action.solar_model, "lux_20260117")
    (I_act, sleep_forced) = light_diet_fn(
        action.light_diet, t, latitude, solar_model;
        lon_offset=lon_offset, eye_fraction=eye_fraction,
    )

    if action.sleep_diet == "forced_sleep"
        sleep_forced = true
    elseif action.sleep_diet == "forced_wake"
        sleep_forced = false
    end

    _validate_sleep_forced(sleep_forced)
    I_act = _sanitize_I_act(I_act)
    return _circadian_transition_impl(
        state, I_act, sleep_forced, action.dt, params;
        deterministic_light=deterministic_light,
        sleep_light_fraction=sleep_light_fraction,
        rng=rng,
    )
end

# ================================================================
# Circadian Transition with Extended State and Action
# ================================================================
# This transition function uses CircadianState and CircadianActionExt
# to compute light intensity on-the-fly from:
#   - state.t (current datetime)
#   - state.latitude, state.longitude (geographic location)
#   - action.light_diet (schedule pattern name)
#   - action.solar_model (solar irradiance model name)
#
# Requires: include("light_diet_functions.jl") to be loaded
# ================================================================

"""
    circadian_transition_ext(state::CircadianState, action::CircadianActionExt, params;
                             light_diet_fn::Function=get_light_intensity,
                             timezone_center_lon::Float64=-120.0,
                             stochastic_light::Bool=true,
                             sleep_blocks_light::Bool=true,
                             check_date_window::Bool=false)

Extended circadian transition: computes light from light diet, then [`transition`](@ref).

Default `stochastic_light=true` draws realized illuminance from `TruncNormal(I_prescribed, 0.1 I_prescribed, 0, 10⁴)` lux (Appendix A.4.1).
With `sleep_blocks_light=true`, sleep uses [`EYELID_LIGHT_FRACTION`](@ref) (≈3% eyelid transmission); set `sleep_blocks_light=false` for full light during sleep.
"""
function circadian_transition_ext(state::CircadianState, action::CircadianActionExt, params;
                                   light_diet_fn::Function=get_light_intensity,
                                   timezone_center_lon::Float64=-120.0,
                                   stochastic_light::Bool=true,
                                   sleep_blocks_light::Bool=true,
                                   check_date_window::Bool=false,
                                   rng::AbstractRNG=Random.default_rng())
    return transition(
        state, action, params;
        rng = rng,
        deterministic_light = !stochastic_light,
        sleep_light_fraction = sleep_blocks_light ? EYELID_LIGHT_FRACTION : 1.0,
        light_diet_fn = light_diet_fn,
        timezone_center_lon = timezone_center_lon,
        check_date_window = check_date_window,
    )
end

"""
    circadian_transition_ext_circadian_only(state::CircadianState, action::CircadianActionExt, params;
                                            light_diet_fn::Function=get_light_intensity,
                                            timezone_center_lon::Float64=-120.0,
                                            eye_fraction::Float64=0.10)

Circadian-only extended transition (no H, Vm, Vv, sleep dynamics). Light during sleep is full (no block).
"""
function circadian_transition_ext_circadian_only(state::CircadianState, action::CircadianActionExt, params;
                                                  light_diet_fn::Function=get_light_intensity,
                                                  timezone_center_lon::Float64=-120.0,
                                                  eye_fraction::Float64=0.10,
                                                  rng::AbstractRNG=Random.default_rng())
    return transition(
        state, action, params;
        rng = rng,
        deterministic_light   = true,
        sleep_light_fraction  = 1.0,
        light_diet_fn         = light_diet_fn,
        timezone_center_lon   = timezone_center_lon,
        eye_fraction          = eye_fraction,
    )
end

# ================================================================
# ACTION DATE WINDOW UTILITIES
# ================================================================

"""
    is_within_action_window(state::CircadianState, action::CircadianActionExt) -> Bool

Check if the state's current time is within the action's date window [startdate, stopdate).

Returns true if action.startdate <= state.t < action.stopdate
"""
function is_within_action_window(state::CircadianState, action::CircadianActionExt)
    return state.t >= action.startdate && state.t < action.stopdate
end

"""
    is_within_action_window(t::DateTime, action::CircadianActionExt) -> Bool

Check if a datetime is within the action's date window [startdate, stopdate).
"""
function is_within_action_window(t::DateTime, action::CircadianActionExt)
    return t >= action.startdate && t < action.stopdate
end

"""
    get_action_duration_hours(action::CircadianActionExt) -> Float64

Get the total duration of the action's date window in hours.
"""
function get_action_duration_hours(action::CircadianActionExt)
    duration_ms = Dates.value(action.stopdate - action.startdate)
    duration_ms >= 0 || error("CircadianActionExt requires stopdate >= startdate")
    return duration_ms / (1000.0 * 3600.0)  # ms to hours
end

"""
    get_action_num_steps(action::CircadianActionExt) -> Int

Get the number of time steps needed to cover the action's date window.
"""
function get_action_num_steps(action::CircadianActionExt)
    action.dt > 0 && isfinite(action.dt) || error("CircadianActionExt.dt must be finite and positive, got $(action.dt)")
    duration_hours = get_action_duration_hours(action)
    n = duration_hours / action.dt
    return n <= 0 ? 0 : ceil(Int, n)
end

"""
    create_action_sequence(actions::Vector{CircadianActionExt}) -> Vector{CircadianActionExt}

Validate and sort a sequence of actions by startdate.
Warns if there are gaps or overlaps in the date windows.
"""
function create_action_sequence(actions::Vector{CircadianActionExt})
    if isempty(actions)
        return actions
    end
    
    # Sort by startdate
    sorted = sort(actions, by = a -> a.startdate)
    
    # Check for gaps and overlaps
    for i in 1:(length(sorted) - 1)
        current_end = sorted[i].stopdate
        next_start = sorted[i+1].startdate
        
        if current_end < next_start
            gap_hours = Dates.value(next_start - current_end) / (1000.0 * 3600.0)
            @warn "Gap of $(gap_hours) hours between actions $(i) and $(i+1)"
        elseif current_end > next_start
            overlap_hours = Dates.value(current_end - next_start) / (1000.0 * 3600.0)
            @warn "Overlap of $(overlap_hours) hours between actions $(i) and $(i+1)"
        end
    end
    
    return sorted
end

"""
    get_current_action(t::DateTime, actions::Vector{CircadianActionExt}) -> Union{CircadianActionExt, Nothing}

Find the action whose date window contains the given time.
Returns nothing if no action covers the time.
"""
function get_current_action(t::DateTime, actions::Vector{CircadianActionExt})
    for action in actions
        if is_within_action_window(t, action)
            return action
        end
    end
    return nothing
end



# ================================================================
# Sleep-Aware Circadian Transition (legacy wrapper)
# ================================================================

"""
    circadian_transition_sleep(state::CircadianState, action::CircadianAction, params;
                              sleep_light_fraction::Float64=EYELID_LIGHT_FRACTION)

Sleep-aware transition: deterministic prescribed illuminance; during sleep, effective light is
`sleep_light_fraction` × prescribed lux (default [`EYELID_LIGHT_FRACTION`](@ref), ~3% eyelid transmission).
Pass `sleep_light_fraction=0.0` to block light completely during sleep.
"""
function circadian_transition_sleep(state::CircadianState, action::CircadianAction, params; rng::AbstractRNG=Random.default_rng(),
                                    sleep_light_fraction::Float64=EYELID_LIGHT_FRACTION)
    _validate_params(params)
    _validate_dt(action.dt)
    _validate_sleep_forced(action.sleep_forced)
    I_act = _sanitize_I_act(something(action.I_act, 0.0))
    return _circadian_transition_impl(
        state, I_act, action.sleep_forced, action.dt, params;
        deterministic_light=true,
        sleep_light_fraction=sleep_light_fraction,
        rng=rng,
    )
end
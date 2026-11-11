# ================================================================
# Circadian POMDP Model Definition
# ================================================================
# Core types and model parameters for the Circadian POMDP.
# Use abstract types for API boundaries; concrete types for storage and dynamics.

# ================================================================
# ABSTRACT TYPES (API boundaries)
# ================================================================
# Subtype these to extend state/action/observation spaces while keeping
# transition and simulation code generic.

"""Abstract supertype for all circadian state representations."""
abstract type AbstractCircadianState end

"""Abstract supertype for actions (precomputed light vs on-the-fly light diet)."""
abstract type AbstractCircadianAction end

"""Abstract supertype for observations (light, sleep events, CBTmin, etc.)."""
abstract type AbstractCircadianObservation end

# ================================================================
# STATE SPACE DEFINITION
# ================================================================

struct CircadianState <: AbstractCircadianState  
    n::Float64         # light sensitivity
    x::Float64         # oscillator x
    xc::Float64        # oscillator xc
    H::Float64         # homeostatic pressure
    Vm::Float64        # MA neuron pop
    Vv::Float64        # VLPO neuron pop
    sleep::Bool        # awake or asleep
    H_plus::Float64    # homeostatic pressure plus
    H_minus::Float64   # homeostatic pressure minus
    tau_x::Float64     # circadian oscillator period
    K::Float64         # sensitivity to light
    chi::Float64       # homeostatic decay parameter
    mu_h::Float64      # homeostatic build up parameter
    phi::Float64       # phase lag of the circadian sleep drive behind the oscillator (h)
    I_eye::Float64     # light intensity from eyes
    latitude::Float64  # latitude
    longitude::Float64 # longitude
    timezone::String   # timezone
    t::DateTime        # time
end

# ================================================================
# ACTION SPACE DEFINITIONS
# ================================================================
# CircadianAction: precomputed light (I_act, sleep_forced, dt).
# CircadianActionExt: on-the-fly light from diet name and state time/location.

struct CircadianAction <: AbstractCircadianAction
    I_act::Union{Float64, Missing}     # applied light
    sleep_forced::Union{Bool, Missing} # forced sleep or wake
    dt::Float64                        # hours
end

struct CircadianActionExt <: AbstractCircadianAction
    light_diet::String                    # pattern of light exposure (e.g. Regular_V4_061hEarlier)
    sleep_diet::String                    # forced sleep or wake
    solar_model::Union{String, Missing}   # solar irradiance model; use missing for clock-based diets that ignore it
    dt::Float64                           # hours
    startdate::DateTime                   # start time
    stopdate::DateTime                    # stop time
end

# ================================================================
# OBSERVATION SPACE DEFINITION
# ================================================================

struct CircadianObservation <: AbstractCircadianObservation
    I_obs::Union{Vector{Union{Float64, Missing}}, Missing}  # timeseries startdate→stopdate; values or series can be missing
    sleep_onset::Vector{DateTime}   # set of sleep onset times
    sleep_offset::Vector{DateTime}  # set of sleep offset times
    DLMO::Vector{DateTime}         # set of DLMO times
    CBT_min::Vector{DateTime}      # set of CBT minimum times
    startdate::DateTime           # observation window start
    stopdate::DateTime            # observation window stop
end

# ================================================================
# MODEL PARAMETERS
# ================================================================

"""
Returns all model parameters organized by category.

Returns a named tuple with:
- params_C: Circadian parameters
- params_H: Homeostatic parameters  
- params_X: Sleep-circadian interaction parameters
"""
function model_parameters()
    # Circadian parameters
    params_C = Dict(
        :beta => 0.007,        # process L feedback
        :G => 37.0,            # light multiplier
        :I_o => 9500.0,        # light normalizer
        :p => 0.5,             # exponent for light transformation
        :mu_c => 0.13,         # oscillator stiffness
        :q => 1/3,             # oscillator parameter
        :tau_x => 24.2,        # intrinsic period of circadian oscillator
        :K => 0.55             # sensitivity to light
    )

    # Homeostatic parameters
    params_H = Dict(
        :chi => 45.0,          # homeostatic decay rate
        :mu_h => 4.2,          # homeostatic build up rate
        :Q_max => 100.0,       # max firing rate per hour
        :theta_h => 10.0,      # neuronal threshold
        :sigma => 3.0,         # neuronal slope parameter
        :tau_m => 1/360,       # MA time constant (hours)
        :tau_v => 1/360,       # VLPO time constant (hours)
        :v_mv => 1.8,          # MA → VLPO connection strength
        :v_vm => 2.1,          # VLPO → MA connection strength
        :v_vc => 3.37,         # circadian → VLPO connection strength
        :v_vh => 1.01,         # homeostatic → VLPO connection strength
        :A_m => 1.3,           # MA baseline activity
        :A_v => -10.2,         # VLPO baseline activity
    )

    # Sleep-circadian interaction parameters
    params_X = Dict(
        :phi => 0.0,           # unused: the transition reads ψ from the state field `phi`
    )

    return (params_C=params_C, params_H=params_H, params_X=params_X)
end

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

"""
Get the types for the POMDP spaces.
Returns a tuple of (AbstractCircadianState, AbstractCircadianAction, AbstractCircadianObservation)
and their default concrete types (CircadianState, CircadianAction, CircadianObservation).
"""
function get_space_types()
    return (CircadianState, CircadianAction, CircadianObservation)
end

"""
Create a default CircadianState with standard parameter values.
Useful for initialization and testing.
"""
function default_circadian_state()
    return CircadianState(
        0.5,        # n
        -1.22,        # x
        -0.17,        # xc
        11.0,       # H
        1.09,       # Vm
        -10.2,      # Vv
        false,      # sleep
        15.0,       # H_plus
        15.0,       # H_minus
        24.2,       # tau_x
        0.55,       # K
        45.0,       # chi
        4.2,        # mu_h
        0.0,        # phi
        0.0,        # I_eye
        37.4277,        # latitude (Stanford, CA)
        -122.1701,        # longitude (Stanford, CA)
        "PST",      # timezone (Stanford, CA)
        DateTime(2023,1,1,0,0,0) # t
    )
end

"""
Create a default CircadianAction.
Useful for testing and initialization.
"""
function default_circadian_action(light_intensity=0.0, dt=1/60)
    return CircadianAction(light_intensity, missing, dt)
end

"""
Create a default CircadianObservation.
Useful for testing and initialization.
Optional: light_obs (Number, AbstractVector, or missing), startdate, stopdate.
"""
function default_circadian_observation(light_obs=0.0; startdate=DateTime(2023, 1, 1, 0, 0, 0), stopdate=DateTime(2023, 1, 2, 0, 0, 0))
    I = if light_obs === missing
        missing
    elseif isa(light_obs, Number)
        Union{Float64, Missing}[light_obs]
    else
        Vector{Union{Float64, Missing}}(light_obs)
    end
    return CircadianObservation(I, DateTime[], DateTime[], DateTime[], DateTime[], startdate, stopdate)
end

# ================================================================
# PARAMETER ACCESS HELPERS
# ================================================================

"""
Get all circadian parameters as a flat dictionary.
Useful when you need all parameters in one place.
"""
function get_all_parameters_flat()
    params = model_parameters()
    flat_params = Dict{Symbol, Float64}()
    
    # Merge all parameter dictionaries
    merge!(flat_params, params.params_C)
    merge!(flat_params, params.params_H)
    merge!(flat_params, params.params_X)
    
    return flat_params
end

"""
Print a summary of all model parameters.
"""
function print_parameter_summary()
    params = model_parameters()
    
    println("=== Circadian POMDP Model Parameters ===")
    println()
    
    println("Circadian Parameters (Process C & P):")
    for (key, val) in params.params_C
        println("  $key: $val")
    end
    
    println("\nHomeostatic Parameters (Process H):")
    for (key, val) in params.params_H
        println("  $key: $val")
    end
    
    println("\nSleep-Circadian Interaction Parameters:")
    for (key, val) in params.params_X
        println("  $key: $val")
    end
    
    println("\nTotal parameters: $(length(get_all_parameters_flat()))")
end 
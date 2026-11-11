"""
CBTmin phase-crossing threshold (radians). The circadian oscillator phase
`atan(xc, x)` crosses this value at CBTmin. Calibrated from St. Hilaire et al. (2007).
"""
const CBTMIN_PHASE_THRESHOLD = deg2rad(-170.7)

"""
Temporal offset (hours) added after phase-crossing detection to align the
detected event with the true CBTmin timing. Calibrated from St. Hilaire et al. (2007).
"""
const CBTMIN_TEMPORAL_OFFSET_H = 0.97

"""
    cbtmin_phase_crossing(prev_phase, cur_phase) -> Union{Float64, Nothing}

Fraction of a step (0–1) at which the oscillator phase `atan(xc, x)` reaches [`CBTMIN_PHASE_THRESHOLD`](@ref)
between two consecutive phases, or `nothing` if it does not. St. Hilaire et al. (2007): CBTmin = time at which
arctan(xc/x) = −170.7°, plus [`CBTMIN_TEMPORAL_OFFSET_H`](@ref). The phase decreases with time (the state turns
clockwise in the x–xc plane), so the threshold is reached from above; the jump from −π to +π once per cycle is not
a crossing.
"""
function cbtmin_phase_crossing(prev_phase::Real, cur_phase::Real)
    thr = CBTMIN_PHASE_THRESHOLD
    (prev_phase >= thr && cur_phase < thr && prev_phase - cur_phase < π) || return nothing
    return clamp((prev_phase - thr) / (prev_phase - cur_phase), 0.0, 1.0)
end

"""
    get_CBTmin(states::Vector{<:AbstractCircadianState}; dt=1/60) -> Vector{Int}

Detect Core Body Temperature minimum (CBTmin) events from a circadian state trajectory.

CBTmin is identified where the oscillator phase angle `atan(xc, x)` reaches
a calibrated threshold (−170.7°, from the St. Hilaire et al. 2007 model;
see [`cbtmin_phase_crossing`](@ref)), with a 0.97 h temporal offset applied
to align the phase-crossing with the true CBTmin timing.

Returns a vector of **state indices** at which CBTmin occurs. Convert to
hours via `(idx - 1) * dt`.

# Arguments
- `states`: trajectory of circadian states (must have `.x` and `.xc` fields).
- `dt`: time step in hours (default `1/60`, i.e. 1 minute).
"""
function get_CBTmin(states::Vector{<:AbstractCircadianState}; dt=1/60)
    p_ref  = CBTMIN_TEMPORAL_OFFSET_H

    # Calculate phase from oscillator components
    phase = atan.([s.xc for s in states],[s.x for s in states])

    # Steps in which the (decreasing) phase reaches the threshold
    CBTmin_idx = findall(i -> cbtmin_phase_crossing(phase[i], phase[i+1]) !== nothing, 1:length(phase)-1)
    
    # Apply temporal shift
    shift_steps = round(Int, p_ref / dt)
    CBTmin_shifted = CBTmin_idx .+ shift_steps
    
    # Filter to valid indices
    CBTmin_shifted = filter(i -> i > 0 && i <= length(states), CBTmin_shifted)
    
    return CBTmin_shifted
end

"""
    SleepPeriod

A detected sleep episode within a circadian state trajectory.

# Fields
- `onset_idx::Union{Int,Missing}`:    state index of sleep onset.
- `offset_idx::Union{Int,Missing}`:   state index of sleep offset (wake).
- `duration::Union{Float64,Missing}`: episode duration in hours.
- `midpoint_idx::Union{Int,Missing}`: state index at the midpoint of the episode.

Convert any index to hours via `(idx - 1) * dt`.
"""
struct SleepPeriod
    onset_idx::Union{Int,Missing}
    offset_idx::Union{Int,Missing}
    duration::Union{Float64,Missing}
    midpoint_idx::Union{Int,Missing}
end

"""
    get_sleep_periods(states::Vector{<:AbstractCircadianState}; dt=1/60) -> Vector{SleepPeriod}

Detect sleep episodes from a circadian state trajectory by finding transitions
in the `.sleep` field. Each onset is paired with its next offset to form a
[`SleepPeriod`](@ref). Unmatched offsets (e.g. waking at the start of the
recording) are skipped.

# Arguments
- `states`: trajectory of circadian states (must have a `.sleep` Bool field).
- `dt`: time step in hours (default `1/60`, i.e. 1 minute).
"""
function get_sleep_periods(states::Vector{<:AbstractCircadianState}; dt=1/60)
    sleep = [s.sleep for s in states]

    n = length(sleep)

    # Correct transitions:
    onsets  = findall(i -> sleep[i] == false && sleep[i+1] == true, 1:n-1) .+ 1
    offsets = findall(i -> sleep[i] == true  && sleep[i+1] == false, 1:n-1) .+ 1

    sleep_unsorted = []

    i, j = 1, 1
    while i <= length(onsets) && j <= length(offsets)
        if onsets[i] < offsets[j]
            dur = (offsets[j] - onsets[i]) * dt
            mid_idx = round(Int, onsets[i] + (offsets[j] - onsets[i]) / 2)
            push!(sleep_unsorted, SleepPeriod(onsets[i], offsets[j], dur, mid_idx))
            i += 1
            j += 1
        else
            j += 1  # skip unmatched offset
        end
    end

    sleeps = sort(sleep_unsorted; by = s -> something(s.onset_idx, -1))

    return sleeps
end

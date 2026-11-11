using Statistics

# ============================================================================
# 24-hour cyclic time utilities
# ============================================================================

"""
    unwrap24(x::AbstractVector{<:Real}) -> Vector{Float64}

Unwrap 24-hour cyclic data to produce a continuous signal by removing
discontinuities larger than 12 hours. Analogous to phase unwrapping for
circular data on a [0, 24) domain.

# Example
```julia
unwrap24([23.0, 23.5, 0.5, 1.0])   # → [23.0, 23.5, 24.5, 25.0]
unwrap24([1.0, 0.5, 23.5, 23.0])   # → [1.0, 0.5, -0.5, -1.0]
```
"""
function unwrap24(x::AbstractVector{<:Real})
    out = Float64.(copy(x))
    for i in 2:length(out)
        d = out[i] - out[i-1]
        if d > 12
            out[i:end] .-= 24
        elseif d < -12
            out[i:end] .+= 24
        end
    end
    return out
end

"""
    wrap24(x::Real) -> Float64

Wrap a value into the [0, 24) range via `mod(x, 24.0)`.
"""
wrap24(x::Real) = mod(Float64(x), 24.0)

# ============================================================================
# Steady-state detection
# ============================================================================

"""
    is_steady_state_window(values::AbstractVector{<:Real}; threshold::Float64=0.25) -> Bool

Return `true` if `values` (in hours, 0–24) are consistent with a steady state:
the range of the unwrapped series is ≤ `threshold` *or* its standard deviation
is ≤ `threshold / 2`.  Requires at least 2 values.
"""
function is_steady_state_window(values::AbstractVector{<:Real}; threshold::Float64=0.25)
    length(values) < 2 && return false
    uw = unwrap24(values)
    return (maximum(uw) - minimum(uw)) <= threshold || std(uw) <= threshold / 2
end

"""
    find_baseline_steady_state_cbtmin(trajectory::AbstractVector{<:Real};
                                      min_consecutive_days::Int=7,
                                      threshold::Float64=0.25) -> Union{Float64, Nothing}

Search the tail of a baseline CBTmin trajectory (one value per circadian day)
for the first window of `min_consecutive_days` values that satisfies
[`is_steady_state_window`](@ref).  Returns the unwrapped mean CBTmin in that
window, or `nothing` if no steady state is found.

The search begins up to 13 days before the last possible window start so that
longer baselines still find a window near the end.
"""
function find_baseline_steady_state_cbtmin(trajectory::AbstractVector{<:Real};
                                           min_consecutive_days::Int=7,
                                           threshold::Float64=0.25)
    n = length(trajectory)
    n < min_consecutive_days && return nothing

    for start_idx in max(1, n - min_consecutive_days - 13):(n - min_consecutive_days + 1)
        end_idx = start_idx + min_consecutive_days - 1
        end_idx > n && break
        window = trajectory[start_idx:end_idx]
        if is_steady_state_window(window; threshold)
            return mean(unwrap24(window))
        end
    end
    return nothing
end

"""
    load_baseline_steady_state_cbtmin(path; baseline_hours, min_consecutive_days=7, threshold=0.25)

Load regular-schedule particle histories from path, extract baseline CBTmin trajectories,
and compute steady-state CBTmin for each particle. Returns (baseline_steady_state_cbtmins, baseline_counts).
"""
function load_baseline_steady_state_cbtmin(path::AbstractString; baseline_hours::Real,
                                           min_consecutive_days::Int=7,
                                           threshold::Float64=0.25)
    hist = load_particle_histories(path)
    trajectories, counts = baseline_cbtmin_from_events(
        hist.cbtmin_events, hist.num_particles, Float64(baseline_hours))
    baseline_cbtmins = Vector{Union{Float64, Nothing}}(undef, hist.num_particles)
    for i in 1:hist.num_particles
        if counts[i] >= min_consecutive_days
            baseline_cbtmins[i] = find_baseline_steady_state_cbtmin(trajectories[i];
                min_consecutive_days=min_consecutive_days, threshold=threshold)
        else
            baseline_cbtmins[i] = nothing
        end
    end
    return baseline_cbtmins, counts
end

"""
    find_first_steady_state_day(trajectory, baseline_cbtmin, tz_shift, baseline_count;
                                min_consecutive_days=7, threshold=0.25,
                                expected_tolerance=0.25)
        -> (days_after_shift, steady_mean, direction)

Given a full CBTmin trajectory (baseline + post-shift), find the first
post-shift day when the trajectory reaches steady state in the new timezone.

*Day 0* is defined as the last baseline CBTmin event (index `baseline_count`);
*day 1* is the first event in the new timezone.

# Returns
A 3-tuple `(days_after_shift::Int, steady_mean::Float64, direction::Symbol)`
where `direction` is `:forward` or `:backward`, or `(nothing, nothing, nothing)`
if no qualifying steady state is found.
"""
function find_first_steady_state_day(trajectory::AbstractVector{<:Real},
                                     baseline_cbtmin::Real,
                                     tz_shift::Real,
                                     baseline_count::Int;
                                     min_consecutive_days::Int=7,
                                     threshold::Float64=0.25,
                                     expected_tolerance::Float64=0.25)
    n = length(trajectory)
    day_zero_idx = baseline_count
    first_new_idx = baseline_count + 1

    if n < first_new_idx + min_consecutive_days - 1
        return (nothing, nothing, nothing)
    end

    expected_cbtmin = wrap24(baseline_cbtmin + tz_shift)

    for start_idx in first_new_idx:(n - min_consecutive_days + 1)
        window = trajectory[start_idx:(start_idx + min_consecutive_days - 1)]
        if !is_steady_state_window(window; threshold)
            continue
        end

        steady_mean = mean(unwrap24(window))
        steady_wrapped = wrap24(steady_mean)

        diff = steady_wrapped - expected_cbtmin
        diff > 12 && (diff -= 24)
        diff < -12 && (diff += 24)
        abs(diff) > expected_tolerance && continue

        # Determine adaptation direction from transition trajectory
        tw_start = max(1, day_zero_idx - 3)
        tw_end = min(n, start_idx + 5)
        tw = unwrap24(trajectory[tw_start:tw_end])

        baseline_idx_in_window = day_zero_idx - tw_start + 1
        if 1 <= baseline_idx_in_window <= length(tw)
            direction = tw[end] - tw[baseline_idx_in_window] > 0 ? :forward : :backward
        else
            fwd = mod(steady_wrapped - wrap24(baseline_cbtmin) + 24, 24)
            direction = fwd <= 12 ? :forward : :backward
        end

        return (start_idx - day_zero_idx, steady_mean, direction)
    end

    return (nothing, nothing, nothing)
end

"""
    compute_adaptation_time(cbtmin_circadian, baseline_cbtmin, baseline_count, tz_shift;
                            min_consecutive_days=7, threshold=0.25,
                            expected_tolerance=0.25)
        -> Union{NamedTuple, Nothing}

Convenience wrapper around [`find_first_steady_state_day`](@ref) that returns
a named tuple `(time, direction, baseline_cbtmin, steady_cbtmin)` or `nothing`.

Unlike the experiment-level helper in `rescuingJetLag_unified.jl`, this function
takes explicit baseline data instead of reading from global state.
"""
function compute_adaptation_time(cbtmin_circadian::AbstractVector{<:Real},
                                 baseline_cbtmin::Real,
                                 baseline_count::Int,
                                 tz_shift::Real;
                                 min_consecutive_days::Int=7,
                                 threshold::Float64=0.25,
                                 expected_tolerance::Float64=0.25)
    days, steady_val, dir = find_first_steady_state_day(
        cbtmin_circadian, baseline_cbtmin, tz_shift, baseline_count;
        min_consecutive_days, threshold, expected_tolerance)

    days === nothing && return nothing

    return (time=days, direction=dir, baseline_cbtmin=Float64(baseline_cbtmin), steady_cbtmin=steady_val)
end

# ============================================================================
# Time normalization (center on midnight)
# ============================================================================

"""
    normalize_time!(times::Vector{Float64}) -> Vector{Float64}

Shift clock-hour values ≥ 12 by −24 so that the result lies in [−12, 12),
centering the distribution around midnight. Mutates and returns `times`.

# Example
```julia
normalize_time!([0.5, 14.0, 22.0])  # → [0.5, -10.0, -2.0]
```
"""
function normalize_time!(times::Vector{Float64})
    times[times .>= 12] .-= 24
    return times
end

"""
    normalize_time(times) -> Vector{Float64}

Non-mutating version of [`normalize_time!`](@ref). Returns a shifted copy.
"""
function normalize_time(times)
    out = Float64.(copy(times))
    normalize_time!(out)
    return out
end

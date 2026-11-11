# ================================================================
# MLE-Based Particle Summary Functions
# ================================================================
# Functions for computing Maximum Likelihood Estimation (MLE) based 
# summaries of particle simulation results, including sleep metrics
# and circadian rhythm metrics.

using KernelDensity
using Statistics

# ================================================================
# HELPERS
# ================================================================

"""
    phase_offset_wrap24(sleep_midpoint_mle, cbtmin_mle) -> Float64

Wrap the phase-angle difference `sleep_midpoint_mle - cbtmin_mle` into the
range [-12, 12), matching the manuscript's ψ_obs = wrap24(S_mp − t_CBTmin_MLE)
definition (S1 Appendix Eq 49), up to the boundary convention inherent to
`mod` (which returns values in [0, 24), so a difference of exactly +12h
wraps to -12h rather than +12h — a measure-zero case for continuous
MLE-estimated phase differences).
"""
phase_offset_wrap24(sleep_midpoint_mle, cbtmin_mle) =
    mod((sleep_midpoint_mle - cbtmin_mle) + 12.0, 24.0) - 12.0

# ================================================================
# CORE MLE SUMMARY FUNCTIONS (events-based only)
# ================================================================
# Event format: (particle_id, step_idx, clock_h, t_hours, datetime)

"""
    compute_mle_particle_summary(cbtmin_events, sleep_onset_events, sleep_offset_events,
        particle_id, initial_particle; dt=1/60, start_day=20.333, min_sleep_episodes=13, time_shift=12.0)

Compute MLE summary from simulation events (no full state history needed).
"""
function compute_mle_particle_summary(
    cbtmin_events, sleep_onset_events, sleep_offset_events,
    particle_id::Int, initial_particle;
    dt=1/60, start_day=20.333, min_sleep_episodes=13, time_shift=12.0
)
    start_t_hours = start_day * 24
    pid = particle_id

    cbtmin_p = [e for e in cbtmin_events if e.particle_id == pid]
    onsets_p = sort([e for e in sleep_onset_events if e.particle_id == pid], by=e->e.t_hours)
    offsets_p = sort([e for e in sleep_offset_events if e.particle_id == pid], by=e->e.t_hours)

    try
        # CBTmin: filter by t_hours >= start_day*24
        CBTmin_times = [((e.clock_h + time_shift) % 24) for e in cbtmin_p if e.t_hours >= start_t_hours]
        if isempty(CBTmin_times)
            return nothing
        end
        KD_CBTmin = KernelDensity.kde(Float64.(CBTmin_times))
        CBTmin_mle = mod(KD_CBTmin.x[argmax(KD_CBTmin.density)] - time_shift, 24)

        # Match onsets with offsets; skip onsets before the analysis window
        considered_sleeps = _pair_onset_offset_events(onsets_p, offsets_p; start_t_hours=start_t_hours)
        if length(considered_sleeps) < min_sleep_episodes
            return nothing
        end
        if length(considered_sleeps) > min_sleep_episodes
            considered_sleeps = considered_sleeps[end-min_sleep_episodes+1:end]
        end

        sleep_durations = [s.offset.t_hours - s.onset.t_hours for s in considered_sleeps]
        sleep_onsets = [((s.onset.t_hours - time_shift) % 24) for s in considered_sleeps]
        sleep_offsets = [((s.offset.t_hours - time_shift) % 24) for s in considered_sleeps]
        sleep_midpoints = [((s.onset.t_hours + s.offset.t_hours)/2 - time_shift) % 24 for s in considered_sleeps]

        KD_duration = KernelDensity.kde(Float64.(sleep_durations))
        KD_onsets = KernelDensity.kde(Float64.(sleep_onsets))
        KD_offsets = KernelDensity.kde(Float64.(sleep_offsets))
        KD_midpoints = KernelDensity.kde(Float64.(sleep_midpoints))

        sleep_duration_mle = KD_duration.x[argmax(KD_duration.density)]
        sleep_onset_mle = mod(KD_onsets.x[argmax(KD_onsets.density)] + time_shift, 24)
        sleep_offset_mle = mod(KD_offsets.x[argmax(KD_offsets.density)] + time_shift, 24)
        sleep_midpoint_mle = mod(KD_midpoints.x[argmax(KD_midpoints.density)] + time_shift, 24)
        phase_offset = phase_offset_wrap24(sleep_midpoint_mle, CBTmin_mle)

        start_idx = round(Int, start_t_hours / dt)
        tau = initial_particle.tau_x
        K = initial_particle.K
        chi = initial_particle.chi
        mu_h = initial_particle.mu_h
        phi = initial_particle.phi

        return (
            CBTmin_mle = CBTmin_mle,
            sleep_onset_mle = sleep_onset_mle,
            sleep_offset_mle = sleep_offset_mle,
            sleep_midpoint_mle = sleep_midpoint_mle,
            sleep_duration_mle = sleep_duration_mle,
            phase_offset = phase_offset,
            tau = tau, K = K, chi = chi, mu_h = mu_h, phi = phi,
            sleep_episodes_count = length(considered_sleeps),
            analysis_start_idx = start_idx,
            sleep_onsets_raw = sleep_onsets,
            sleep_offsets_raw = sleep_offsets,
            sleep_midpoints_raw = sleep_midpoints,
            sleep_durations_raw = sleep_durations,
            cbtmin_times_raw = CBTmin_times
        )
    catch e
        @warn "Error computing MLE summary for particle $pid: $e"
        return nothing
    end
end

"""
    compute_mle_population_summaries(cbtmin_events, sleep_onset_events, sleep_offset_events,
        initial_particles; verbose=true, kwargs...)

Compute MLE summaries from simulation events.
"""
function compute_mle_population_summaries(
    cbtmin_events, sleep_onset_events, sleep_offset_events,
    initial_particles; verbose=true, kwargs...
)
    summaries = []
    N = length(initial_particles)
    if verbose
        println("Computing MLE summaries for $N particles...")
    end
    for i in 1:N
        if verbose && i % 100 == 0
            println("Processing particle $i/$N...")
        end
        s = compute_mle_particle_summary(
            cbtmin_events, sleep_onset_events, sleep_offset_events,
            i, initial_particles[i]; kwargs...
        )
        if s !== nothing
            push!(summaries, merge(s, (particle_id = i,)))
        end
    end
    if verbose
        println("Successfully computed $(length(summaries))/$N summaries")
    end
    return summaries
end

# ================================================================
# ANALYSIS AND STATISTICS FUNCTIONS
# ================================================================

"""
    analyze_mle_summaries(summaries)

Analyze a collection of MLE summaries to extract population statistics.

# Arguments
- `summaries`: Vector of MLE summary tuples

# Returns
Dictionary with population-level statistics
"""
function analyze_mle_summaries(summaries)
    if isempty(summaries)
        return Dict("error" => "No summaries provided")
    end
    
    # Extract fields for analysis
    cbtmin_mles = [s.CBTmin_mle for s in summaries]
    sleep_onset_mles = [s.sleep_onset_mle for s in summaries]
    sleep_offset_mles = [s.sleep_offset_mle for s in summaries]
    sleep_midpoint_mles = [s.sleep_midpoint_mle for s in summaries]
    sleep_duration_mles = [s.sleep_duration_mle for s in summaries]
    phase_offsets = [s.phase_offset for s in summaries]
    
    # Model parameters
    taus = [s.tau for s in summaries]
    Ks = [s.K for s in summaries]
    chis = [s.chi for s in summaries]
    mu_hs = [s.mu_h for s in summaries]
    phis = [s.phi for s in summaries]
    
    return Dict(
        "population_size" => length(summaries),
        "CBTmin_mean" => mean(cbtmin_mles),
        "CBTmin_std" => std(cbtmin_mles),
        "sleep_onset_mean" => mean(sleep_onset_mles),
        "sleep_onset_std" => std(sleep_onset_mles),
        "sleep_offset_mean" => mean(sleep_offset_mles),
        "sleep_offset_std" => std(sleep_offset_mles),
        "sleep_midpoint_mean" => mean(sleep_midpoint_mles),
        "sleep_midpoint_std" => std(sleep_midpoint_mles),
        "sleep_duration_mean" => mean(sleep_duration_mles),
        "sleep_duration_std" => std(sleep_duration_mles),
        "phase_offset_mean" => mean(phase_offsets),
        "phase_offset_std" => std(phase_offsets),
        "tau_mean" => mean(taus),
        "tau_std" => std(taus),
        "K_mean" => mean(Ks),
        "K_std" => std(Ks),
        "chi_mean" => mean(chis),
        "chi_std" => std(chis),
        "mu_h_mean" => mean(mu_hs),
        "mu_h_std" => std(mu_hs),
        "phi_mean" => mean(phis),
        "phi_std" => std(phis)
    )
end

"""
    print_mle_summary_stats(analysis_results)

Print formatted summary statistics from MLE analysis.
"""
function print_mle_summary_stats(analysis_results)
    println("=== MLE Population Summary Statistics ===")
    println("Population size: $(analysis_results["population_size"])")
    println()
    
    println("Circadian Metrics:")
    println("  CBTmin: $(round(analysis_results["CBTmin_mean"], digits=2)) ± $(round(analysis_results["CBTmin_std"], digits=2)) hours")
    println()
    
    println("Sleep Metrics:")
    println("  Sleep onset: $(round(analysis_results["sleep_onset_mean"], digits=2)) ± $(round(analysis_results["sleep_onset_std"], digits=2)) hours")
    println("  Sleep offset: $(round(analysis_results["sleep_offset_mean"], digits=2)) ± $(round(analysis_results["sleep_offset_std"], digits=2)) hours")
    println("  Sleep midpoint: $(round(analysis_results["sleep_midpoint_mean"], digits=2)) ± $(round(analysis_results["sleep_midpoint_std"], digits=2)) hours")
    println("  Sleep duration: $(round(analysis_results["sleep_duration_mean"], digits=2)) ± $(round(analysis_results["sleep_duration_std"], digits=2)) hours")
    println("  Phase offset: $(round(analysis_results["phase_offset_mean"], digits=2)) ± $(round(analysis_results["phase_offset_std"], digits=2)) hours")
    println()
    
    println("Model Parameters:")
    println("  τ (period): $(round(analysis_results["tau_mean"], digits=2)) ± $(round(analysis_results["tau_std"], digits=2)) hours")
    println("  K (light sensitivity): $(round(analysis_results["K_mean"], digits=3)) ± $(round(analysis_results["K_std"], digits=3))")
    println("  χ (homeostatic decay): $(round(analysis_results["chi_mean"], digits=1)) ± $(round(analysis_results["chi_std"], digits=1))")
    println("  μₕ (homeostatic buildup): $(round(analysis_results["mu_h_mean"], digits=2)) ± $(round(analysis_results["mu_h_std"], digits=2))")
    println("  ψ (circadian drive phase lag, h): $(round(analysis_results["phi_mean"], digits=3)) ± $(round(analysis_results["phi_std"], digits=3))")
end

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

"""
    save_mle_summaries(summaries, filepath; include_analysis=true, verbose=true)

Save MLE summaries to a JLD2 file with optional analysis results.

# Arguments
- `summaries`: Vector of MLE summary tuples
- `filepath`: Path where to save the file
- `include_analysis`: Whether to include population analysis (default: true)
- `verbose`: Whether to print save confirmation (default: true)
"""
function save_mle_summaries(summaries, filepath; include_analysis=true, verbose=true)
    save_data = Dict("summaries" => summaries)
    
    if include_analysis && !isempty(summaries)
        analysis = analyze_mle_summaries(summaries)
        save_data["analysis"] = analysis
    end
    
    mkpath(dirname(filepath))  # Create directory if needed
    
    JLD2.save(filepath, save_data)
    
    if verbose
        println("Saved $(length(summaries)) MLE summaries to $filepath")
        if include_analysis
            println("Analysis results included in saved file")
        end
    end
end

"""
    load_mle_summaries(filepath)

Load MLE summaries from a JLD2 file.

# Returns
Named tuple with summaries and analysis (if available)
"""
function load_mle_summaries(filepath)
    data = JLD2.load(filepath)
    
    summaries = data["summaries"]
    analysis = haskey(data, "analysis") ? data["analysis"] : nothing
    
    return (summaries=summaries, analysis=analysis)
end

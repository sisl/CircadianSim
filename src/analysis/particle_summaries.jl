"""
    plot_summary_histograms(summaries)

Plot histograms of numeric fields from an array of summary NamedTuples.

!!! warning "Deprecated"
    This function predates the MLE summary pipeline. Consider using
    `compute_mle_population_summaries` or `compute_mle_population_summaries_from_events`
    for newer analyses.
"""
function plot_summary_histograms(summaries)
    # Get field names from the first summary
    fields = propertynames(summaries[1])

    # Determine which fields are numeric
    numeric_fields = [f for f in fields if eltype(getfield.(summaries, f)) <: Number]

    # Plot histograms for each numeric field
    n = length(numeric_fields)
    ncols = min(n, 4)
    nrows = ceil(Int, n / ncols)

    plot_list = []

    for f in numeric_fields
        values = getfield.(summaries, f)
        p = Plots.histogram(values, bins=100, title=string(f), legend=false)
        push!(plot_list, p)
    end

    # Combine plots into a grid layout
    return Plots.plot(plot_list..., layout=(nrows, ncols), size=(1000, 300 * nrows))
end

"""
    estimate_VonMeises(phases_rad)

Estimate the mean direction (in hours), resultant length R̄, and concentration
parameter κ of a von Mises distribution from phase angles (radians on [0, 2π]).

!!! warning "Deprecated"
    Use `circular_mean_clock_h` for circular mean computation in newer code.

# Returns
`(mu, R̄, κ)` — mean direction (hours), resultant length, concentration.
"""
function estimate_VonMeises(phases_rad)
    C = mean(cos.(phases_rad))
    S = mean(sin.(phases_rad))
    mu = 24/(2*pi)*atan(S, C)       # mean direction (radians), in [-π, π]
    R̄ = sqrt(C^2 + S^2) 
    if R̄ < 0.53
        K = 2 * R̄ + R̄^3 + (5 * R̄^5) / 6
    elseif R̄ < 0.85
        K = -0.4 + 1.39 * R̄ + 0.43 / (1 - R̄)
    else
        K = 1 / (R̄^3 - 4 * R̄^2 + 3 * R̄)
    end
    return mu, R̄, K   
end

"""
    particle_summaries(particle_histories, t)

Compute per-particle MLE summaries (CBTmin, sleep onset/offset/midpoint/duration)
from full state histories using KDE mode estimation.

!!! warning "Deprecated"
    Use `compute_mle_population_summaries` or
    `compute_mle_population_summaries_from_events` instead.
"""
function particle_summaries(particle_histories,t)

    t_24_rad = 2*pi/24 .* (t .% 24)

    summaries = []

    for i in eachindex(particle_histories)

        try
            # === Circadian rhythm metrics ===
            # extract CBTmin metrics
            CBTmin_idx = get_CBTmin(particle_histories[i],dt=1/60)
            CBTmin = [(t[j] .% 24) for j in CBTmin_idx]

            # KDE
            KD = KernelDensity.kde(Float64.(CBTmin))
            CBTmin_mle = KD.x[argmax(KD.density)]

            # === Sleep metrics ===
            # Extract sleep metrics
            sleep = [s.sleep for s in particle_histories[i]]
            sleeps = get_sleep_periods(particle_histories[i]; dt=1/60)

            onsets = [s.onset_idx for s in sleeps]
            offsets = [s.offset_idx for s in sleeps]
            midpoints = [s.midpoint_idx for s in sleeps]
            
            sleep_onset = [(t[j] .% 24) for j in onsets]
            sleep_offset = [(t[j] .% 24) for j in offsets]
            sleep_midpoint = [(t[j] .% 24) for j in midpoints]

            # KDE
            KD_sOn = KernelDensity.kde(Float64.(sleep_onset))
            KD_sOff = KernelDensity.kde(Float64.(sleep_offset))
            KD_sMid = KernelDensity.kde(Float64.(sleep_midpoint))

            sleep_onset_mle = KD_sOn.x[argmax(KD_sOn.density)]
            sleep_offset_mle = KD_sOff.x[argmax(KD_sOff.density)]
            sleep_midpoint_mle = KD_sMid.x[argmax(KD_sMid.density)]

            # Sleep Duration Metrics
            sleep_duration = [s.duration for s in sleeps]
            KD_sDur = KernelDensity.kde(Float64.(sleep_duration))
            sleep_duration_mle = KD_sDur.x[argmax(KD_sDur.density)]

            # === Model parameters ===
            tau = particle_histories[i][1].tau_x
            K = particle_histories[i][1].K
            chi = particle_histories[i][1].chi
            mu_h = particle_histories[i][1].mu_h
            phi = particle_histories[i][1].phi

            summary = (
                    CBTmin_mle = CBTmin_mle,
                    sleep_onset_mle = sleep_onset_mle,
                    sleep_offset_mle = sleep_offset_mle,
                    sleep_midpoint_mle = sleep_midpoint_mle,
                    sleep_duration_mle = sleep_duration_mle,
                    tau = tau,
                    K = K,
                    chi = chi,
                    mu_h = mu_h,
                    phi = phi,
                    particle_id = i
                )

                push!(summaries, summary)
        catch e
            @warn "Failed to compute summary for particle $i" exception=e
        end
    end

    return summaries
end
# Summarize Jet Lag Adaptation Statistics
# Lara Weed
# For each timezone shift, summarizes:
# - Number of particles that phase advance vs phase delay
# - Adaptation time statistics for each direction
# - Overall adaptation time statistics

using CircadianPOMDP  # load before JLD2 so CircadianState exists when loading initial_particles
using JLD2
using Statistics
using StatsBase
using CSV
using DataFrames
using Plots
using PyCall

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

# ================================================================
# CONFIGURATION (from jetlag_reference.jl)
# ================================================================
# Condition: HighLight10000lux (default) or LowLight150lux. Pass via ARGS: julia summarizeAdaptationStatistics.jl LowLight150lux

const CONDITION = length(ARGS) >= 1 ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT

const OUTPUT_DIR = jetlag_analysis_dir(condition=CONDITION)
const STEADY_STATE_RESULTS_PATH = steady_state_results_path(condition=CONDITION)
const SHAP_CACHE_PATH = jetlag_shap_cache_path(condition=CONDITION)   # v2: joined by particle_id (#27)

# SHAP feature names, colors, line styles, and markers for grayscale compatibility
const SHAP_FEATURE_NAMES = ["τ", "k", "ψ", "χ", "μ_H/χ", 
                            "CBTmin", "Sleep Onset", "Sleep Offset", "Phase Offset", "Sleep Duration"]
const SHAP_FEATURE_COLORS = [:blue, :red, :green, :orange, :purple,
                             :cyan, :magenta, :darkgoldenrod, :brown, :deeppink]
# Grayscale-compatible: line styles and marker shapes
const SHAP_FEATURE_LINESTYLES = [:solid, :dash, :dot, :dashdot, :solid, 
                                  :dash, :dot, :dashdot, :solid, :dash]
const SHAP_FEATURE_MARKERS = [:circle, :square, :diamond, :utriangle, :dtriangle,
                              :circle, :square, :diamond, :utriangle, :dtriangle]
# Grayscale colors (as tuples for Plots.jl compatibility)
# Minimum fraction of particles in a direction (delay vs advance) to show adaptation-time stats
const MIN_ADAPTATION_PLOT_FRACTION = 0.05

const SHAP_FEATURE_GRAYSCALE = [(0.0, 0.0, 0.0),      # black
                                (0.4, 0.4, 0.4),      # dark gray
                                (0.6, 0.6, 0.6),      # medium gray
                                (0.2, 0.2, 0.2),      # very dark gray
                                (0.8, 0.8, 0.8),      # light gray
                                (0.0, 0.0, 0.0),      # black (repeat pattern)
                                (0.4, 0.4, 0.4),      # dark gray
                                (0.6, 0.6, 0.6),      # medium gray
                                (0.2, 0.2, 0.2),      # very dark gray
                                (0.8, 0.8, 0.8)]      # light gray

# ================================================================
# LOAD DATA
# ================================================================

println("📊 Summarizing Jet Lag Adaptation Statistics (condition: $CONDITION)")
println("="^70)

if !isfile(STEADY_STATE_RESULTS_PATH)
    error("Steady state detection results not found: $STEADY_STATE_RESULTS_PATH\nPlease run detectSteadyState.jl first.")
end

results_data = load(STEADY_STATE_RESULTS_PATH)
steady_state_days_all = results_data["steady_state_days_by_timezone"]
steady_state_ids_all = results_data["steady_state_particle_ids_by_timezone"]   # particle id per adaptation time
adaptation_directions_all = results_data["adaptation_directions_by_timezone"]
timezone_shifts = results_data["timezone_shifts"]
# Included particles without a steady state within the post-shift window: counted at the window length (ranked
# slowest) in the overall median and the SHAP fits; delay/advance statistics and percentages cover adapters only
not_adapted_ids_all = results_data["not_adapted_particle_ids_by_timezone"]
const WINDOW_DAYS = results_data["post_shift_window_days"]

println("   ✓ Loaded steady state detection results")
println("   ✓ Processing $(length(timezone_shifts)) timezone shifts")
println()

# ================================================================
# CALCULATE STATISTICS
# ================================================================

# Note: In the code, :forward means CBTmin moved forward (later in time) = phase delay
#       :backward means CBTmin moved backward (earlier in time) = phase advance

summary_rows = []

for (tz_idx, tz_shift) in enumerate(timezone_shifts)
    if tz_idx > length(steady_state_days_all) || isempty(steady_state_days_all[tz_idx])
        continue
    end
    
    adaptation_times = steady_state_days_all[tz_idx]
    directions = adaptation_directions_all[tz_idx]
    
    if length(adaptation_times) != length(directions)
        @warn "Mismatch in lengths for timezone $tz_shift: $(length(adaptation_times)) times vs $(length(directions)) directions"
        continue
    end
    
    # Separate by direction
    # :forward = phase delay (moving later in time)
    # :backward = phase advance (moving earlier in time)
    forward_times = Float64[]
    backward_times = Float64[]
    
    for i in eachindex(adaptation_times)
        if directions[i] == :forward
            push!(forward_times, adaptation_times[i])
        elseif directions[i] == :backward
            push!(backward_times, adaptation_times[i])
        end
    end
    
    # Overall statistics (n_total = adapters; percentages by direction are of adapters)
    n_total = length(adaptation_times)
    n_forward = length(forward_times)
    n_backward = length(backward_times)
    n_not_adapted = length(not_adapted_ids_all[tz_idx])

    # Overall adaptation time statistics: non-adapters counted at the window length (exact while < 25 % are censored)
    censored_times = vcat(Float64.(adaptation_times), fill(Float64(WINDOW_DAYS), n_not_adapted))
    overall_median = median(censored_times)
    overall_q25 = quantile(censored_times, 0.25)
    overall_q75 = quantile(censored_times, 0.75)
    
    # Forward (phase delay) statistics
    # Note: NaN occurs when there are no particles in this category (e.g., all particles choose the shorter path)
    forward_median = isempty(forward_times) ? NaN : median(forward_times)
    forward_q25 = isempty(forward_times) ? NaN : quantile(forward_times, 0.25)
    forward_q75 = isempty(forward_times) ? NaN : quantile(forward_times, 0.75)
    
    # Backward (phase advance) statistics
    # Note: NaN occurs when there are no particles in this category
    backward_median = isempty(backward_times) ? NaN : median(backward_times)
    backward_q25 = isempty(backward_times) ? NaN : quantile(backward_times, 0.25)
    backward_q75 = isempty(backward_times) ? NaN : quantile(backward_times, 0.75)
    
    # Store results
    push!(summary_rows, (
        timezone_shift = tz_shift,
        n_total = n_total,
        n_not_adapted = n_not_adapted,
        pct_not_adapted = round(100 * n_not_adapted / (n_total + n_not_adapted), digits=1),
        post_shift_window_days = WINDOW_DAYS,
        n_phase_delay = n_forward,
        n_phase_advance = n_backward,
        pct_phase_delay = round(100 * n_forward / n_total, digits=1),
        pct_phase_advance = round(100 * n_backward / n_total, digits=1),
        overall_median = round(overall_median, digits=2),
        overall_q25 = round(overall_q25, digits=2),
        overall_q75 = round(overall_q75, digits=2),
        phase_delay_median = round(forward_median, digits=2),
        phase_delay_q25 = round(forward_q25, digits=2),
        phase_delay_q75 = round(forward_q75, digits=2),
        phase_advance_median = round(backward_median, digits=2),
        phase_advance_q25 = round(backward_q25, digits=2),
        phase_advance_q75 = round(backward_q75, digits=2)
    ))
end

# Convert to DataFrame
summary_df = DataFrame(summary_rows)

# ================================================================
# PRINT SUMMARY
# ================================================================

println("📈 Summary Statistics by Timezone Shift")
println("="^70)
println()

for row in eachrow(summary_df)
    println("Timezone Shift: $(row.timezone_shift) hours")
    println("  Total particles: $(row.n_total)")
    println("  Phase Delay (forward): $(row.n_phase_delay) ($(row.pct_phase_delay)%)")
    if row.n_phase_delay > 0
        println("    Median: $(row.phase_delay_median) days, Q25-Q75: $(row.phase_delay_q25)-$(row.phase_delay_q75) days")
    else
        println("    (No particles in this category - all particles chose phase advance)")
    end
    println("  Phase Advance (backward): $(row.n_phase_advance) ($(row.pct_phase_advance)%)")
    if row.n_phase_advance > 0
        println("    Median: $(row.phase_advance_median) days, Q25-Q75: $(row.phase_advance_q25)-$(row.phase_advance_q75) days")
    else
        println("    (No particles in this category - all particles chose phase delay)")
    end
    println("  Overall:")
    println("    Median: $(row.overall_median) days, Q25-Q75: $(row.overall_q25)-$(row.overall_q75) days")
    println()
end

# ================================================================
# SAVE TO CSV
# ================================================================

mkpath(OUTPUT_DIR)
csv_file = joinpath(OUTPUT_DIR, "adaptation_statistics_summary.csv")
CSV.write(csv_file, summary_df)

println("✅ Summary saved to: $csv_file")
println()

# ================================================================
# CREATE SUMMARY TABLE (FORMATTED)
# ================================================================

println("📋 Formatted Summary Table")
println("="^70)
println()

# Print header
println(lpad("TZ", 4), " | ",
        lpad("Total", 6), " | ",
        lpad("Delay", 6), " ", lpad("%", 5), " | ",
        lpad("Advance", 7), " ", lpad("%", 5), " | ",
        lpad("Overall", 20), " | ",
        lpad("Delay Stats", 20), " | ",
        lpad("Advance Stats", 20))
println("-"^120)

# Print rows
for row in eachrow(summary_df)
    overall_str = "$(row.overall_median) ($(row.overall_q25)-$(row.overall_q75))"
    delay_str = row.n_phase_delay > 0 ? "$(row.phase_delay_median) ($(row.phase_delay_q25)-$(row.phase_delay_q75))" : "N/A (n=0)"
    advance_str = row.n_phase_advance > 0 ? "$(row.phase_advance_median) ($(row.phase_advance_q25)-$(row.phase_advance_q75))" : "N/A (n=0)"
    
    println(lpad(row.timezone_shift, 4), " | ",
            lpad(row.n_total, 6), " | ",
            lpad(row.n_phase_delay, 6), " ", lpad("$(row.pct_phase_delay)%", 5), " | ",
            lpad(row.n_phase_advance, 7), " ", lpad("$(row.pct_phase_advance)%", 5), " | ",
            lpad(overall_str, 20), " | ",
            lpad(delay_str, 20), " | ",
            lpad(advance_str, 20))
end

println()

# ================================================================
# CREATE PLOT
# ================================================================

println("📈 Creating plot of phase delay proportion...")

# Set publication-ready defaults
Plots.default(fontfamily="Times", dpi=300, linewidth=2.5)

# Extract data for plotting
tz_shifts = [row.timezone_shift for row in eachrow(summary_df)]
pct_delay = [row.pct_phase_delay for row in eachrow(summary_df)]

# Extract adaptation time medians and quartiles (25th and 75th percentiles)
# Calculate from individual particle data
delay_medians = Float64[]
delay_q25s = Float64[]  # 25th percentile
delay_q75s = Float64[]  # 75th percentile
advance_medians = Float64[]
advance_q25s = Float64[]  # 25th percentile
advance_q75s = Float64[]  # 75th percentile
valid_tz_for_times = Int[]

for (tz_idx, tz_shift) in enumerate(timezone_shifts)
    if tz_idx > length(steady_state_days_all) || isempty(steady_state_days_all[tz_idx])
        continue
    end
    
    adaptation_times = steady_state_days_all[tz_idx]
    directions = adaptation_directions_all[tz_idx]
    
    if length(adaptation_times) != length(directions)
        continue
    end
    
    push!(valid_tz_for_times, tz_shift)
    
    # Separate by direction
    forward_times = Float64[]
    backward_times = Float64[]
    
    for i in eachindex(adaptation_times)
        if directions[i] == :forward
            push!(forward_times, adaptation_times[i])
        elseif directions[i] == :backward
            push!(backward_times, adaptation_times[i])
        end
    end

    n_total = length(adaptation_times)

    # Calculate statistics for phase delay (forward); omit from plot if <5% of particles
    if !isempty(forward_times) && length(forward_times) / n_total >= MIN_ADAPTATION_PLOT_FRACTION
        push!(delay_medians, median(forward_times))
        push!(delay_q25s, quantile(forward_times, 0.25))
        push!(delay_q75s, quantile(forward_times, 0.75))
    else
        push!(delay_medians, NaN)
        push!(delay_q25s, NaN)
        push!(delay_q75s, NaN)
    end

    # Calculate statistics for phase advance (backward); omit from plot if <5% of particles
    if !isempty(backward_times) && length(backward_times) / n_total >= MIN_ADAPTATION_PLOT_FRACTION
        push!(advance_medians, median(backward_times))
        push!(advance_q25s, quantile(backward_times, 0.25))
        push!(advance_q75s, quantile(backward_times, 0.75))
    else
        push!(advance_medians, NaN)
        push!(advance_q25s, NaN)
        push!(advance_q75s, NaN)
    end
end

# Create subplot layout (2 rows, 1 column)
p1 = plot(size=(1400, 380),
          background_color=:white,
          foreground_color=:black,
          framestyle=:box,
          margin=8Plots.mm,
          bottom_margin=6Plots.mm,
          left_margin=14Plots.mm,
          top_margin=4Plots.mm,
          right_margin=6Plots.mm)

# Plot phase delay proportion (top panel) only
plot!(p1, tz_shifts, pct_delay,
      linewidth=3.5,
      color="#345995",
      linestyle=:solid,
      marker=:circle,
      markersize=7,
      markeralpha=0.8,
      markerstrokewidth=0,
      label="Phase Delay",
      xlabel="",
      ylabel="Proportion (%)",
      title="",
      guidefont=("Times", 18, :bold),
      tickfont=("Times", 12),
      legend=(0.85, 0.75),
      legendfont=("Times", 13),
      grid=true,
      gridwidth=1.2,
      gridalpha=0.2,
      gridcolor=:gray60,
      gridstyle=:dash,
      xlim=(0, 24),
      ylim=(0, 105),
      xticks=(4:4:24, ["" for _ in 4:4:24]),
      linealpha=0.9)

# Add reference line at 50%
hline!(p1, [50],
       linestyle=:dash,
       linewidth=1.5,
       color=:gray40,
       alpha=0.6,
       label="")

# Add vertical line at 12 hours (transition point)
vline!(p1, [12],
       linestyle=:dash,
       linewidth=1.5,
       color=:gray40,
       alpha=0.6,
       label="")

# Middle panel: Adaptation times
p2 = plot(size=(1400, 380),
          background_color=:white,
          foreground_color=:black,
          framestyle=:box,
          margin=8Plots.mm,
          bottom_margin=6Plots.mm,
          left_margin=14Plots.mm,
          top_margin=4Plots.mm,
          right_margin=6Plots.mm)

# Phase delay Q25–Q75 shading (behind lines)
plot!(p2, valid_tz_for_times, delay_q75s,
      fillrange=delay_q25s,
      fillalpha=0.25,
      fillcolor="#345995",
      linealpha=0.0,
      label="")

# Phase advance Q25–Q75 shading
plot!(p2, valid_tz_for_times, advance_q75s,
      fillrange=advance_q25s,
      fillalpha=0.25,
      fillcolor="#EAC435",
      linealpha=0.0,
      label="")

# Phase delay median; points with <5% particles are NaN and omitted
plot!(p2, valid_tz_for_times, delay_medians,
      linewidth=3.5,
      color="#345995",
      marker=:circle,
      markersize=7,
      markeralpha=0.8,
      markerstrokewidth=0,
      linestyle=:solid,
      label="Phase Delay",
      xlabel="",
      ylabel="Adaptation Time (days)",
      title="",
      guidefont=("Times", 18, :bold),
      tickfont=("Times", 12),
      legend=(0.85, 0.75),
      legendfont=("Times", 13),
      grid=true,
      gridwidth=1.2,
      gridalpha=0.2,
      gridcolor=:gray60,
      gridstyle=:dash,
      linealpha=0.9,
      xlim=(0, 24),
      ylim=(0, 50),
      xticks=(4:4:24, ["" for _ in 4:4:24]))

# Phase advance median (dashed); points with <5% particles are NaN and omitted
plot!(p2, valid_tz_for_times, advance_medians,
      linewidth=3.5,
      color="#EAC435",
      marker=:square,
      markersize=7,
      markeralpha=0.8,
      markerstrokewidth=1,
      markerstrokecolor=:gray,
      linestyle=:dash,
      label="Phase Advance",
      linealpha=0.9)

# Add vertical line at 12 hours (transition point)
vline!(p2, [12],
       linestyle=:dash,
       linewidth=1.5,
       color=:gray40,
       alpha=0.6,
       label="")

# ================================================================
# COMPUTE SHAP2 IMPORTANCE FOR TRENDS PLOT
# ================================================================

println("🔍 Computing SHAP2 importance for trends plot...")

# Check for cached SHAP values
importance_by_tz = Dict{Int, Vector{Float64}}()
use_cache = true

if use_cache && isfile(SHAP_CACHE_PATH)
    try
        cache_data = load(SHAP_CACHE_PATH)
        if haskey(cache_data, "importance_by_tz")
            global importance_by_tz = cache_data["importance_by_tz"]
            println("   ✓ Loaded SHAP importance from cache ($(length(importance_by_tz)) timezones)")
        else
            @warn "Cache file exists but missing 'importance_by_tz' key. Recomputing..."
            global use_cache = false
        end
    catch e
        @warn "Error loading SHAP cache: $e\nRecomputing..."
        global use_cache = false
    end
else
    global use_cache = false
end

# Compute SHAP values if not cached
if !use_cache
    # Features per particle from the included regular-schedule summaries (they carry the static parameters
    # and particle_id), joined to adaptation times by particle_id (issue #27: was aligned by row index).
    rs_path = regular_summaries_path(condition=CONDITION)
    if !isfile(rs_path)
        @warn "Regular schedule summaries not found: $rs_path\nSkipping SHAP2 trends plot."
        importance_by_tz = Dict{Int, Vector{Float64}}()
    else
        begin
            summaries = filter_included(load(rs_path)["summaries"])
            features = extract_summary_features(summaries; normalize_time_outputs=true)
            row_of_id = Dict{Int,Vector{Float64}}()
            for (k, s_) in enumerate(summaries)
                row = [features.inputs.tau[k], features.inputs.k[k], features.inputs.psi[k], features.inputs.chi[k],
                       features.inputs.mu_h_over_chi[k], features.outputs.cbtmin[k], features.outputs.s_on[k],
                       features.outputs.s_off[k], features.outputs.psi_obs[k], features.outputs.s_dur[k]]
                all(isfinite, row) && (row_of_id[s_.particle_id] = row)
            end
            n_features = length(SHAP_FEATURE_NAMES)
            println("   Feature rows for $(length(row_of_id)) included particles (joined to adaptation times by particle_id)")

            # Compute SHAP importance for each timezone
            local importance_by_tz_compute = Dict{Int, Vector{Float64}}()
            
            for (tz_idx, tz_shift) in enumerate(timezone_shifts)
                if tz_idx > length(steady_state_days_all) || isempty(steady_state_days_all[tz_idx])
                    continue
                end
                
                # Adapters plus non-adapters at the window length (censored)
                ids_tz = vcat(steady_state_ids_all[tz_idx], not_adapted_ids_all[tz_idx])
                days_tz = vcat(Float64.(steady_state_days_all[tz_idx]), fill(Float64(WINDOW_DAYS), length(not_adapted_ids_all[tz_idx])))
                keep = [i for i in eachindex(ids_tz) if haskey(row_of_id, ids_tz[i]) && isfinite(days_tz[i])]
                length(keep) < 10 && continue
                X_tz_clean = permutedims(hcat([row_of_id[ids_tz[i]] for i in keep]...))
                y_tz_clean = adaptation_shap_y(days_tz[keep])   # log(1 + days) unless SHAP_TARGET=days

                try
                    # Compute SHAP
                    X_py = PyObject(X_tz_clean)
                    y_py = PyObject(y_tz_clean)
                    
                    py"""
                    import numpy as np
                    import shap
                    from sklearn.ensemble import RandomForestRegressor
                    
                    X_data = $X_py
                    y_data = $y_py
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_data, y_data)
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_data)
                    
                    import __main__
                    __main__.shap_values_result = shap_values
                    """
                    
                    shap_vals = Array{Float64}(py"__main__.shap_values_result")
                    mean_abs_shap = vec(mean(abs.(shap_vals), dims=1))
                    
                    importance_by_tz_compute[tz_shift] = mean_abs_shap
                    
                catch e
                    @warn "Error computing SHAP for $(tz_shift)h: $e"
                    continue
                end
            end
            
            importance_by_tz = importance_by_tz_compute
            println("   ✓ Computed SHAP importance for $(length(importance_by_tz)) timezones")
            
            # Save to cache for future runs
            if !isempty(importance_by_tz)
                mkpath(dirname(SHAP_CACHE_PATH))
                jldopen(SHAP_CACHE_PATH, "w") do file
                    file["importance_by_tz"] = importance_by_tz
                end
                println("   ✓ Cached SHAP importance to: $(basename(SHAP_CACHE_PATH))")
            end
        end
    end
end

# ================================================================
# CREATE SHAP2 TRENDS SUBPLOT AND COMBINE PLOTS
# ================================================================

if !isempty(importance_by_tz)
    println("📈 Creating SHAP2 trends subplot...")
    
    valid_tz_shap = sort(collect(keys(importance_by_tz)))
    
    # Bottom panel: SHAP2 trends
    p3 = plot(size=(1400, 380),
              background_color=:white,
              foreground_color=:black,
              framestyle=:box,
              margin=8Plots.mm,
              bottom_margin=10Plots.mm,
              left_margin=14Plots.mm,
              top_margin=4Plots.mm,
              right_margin=6Plots.mm)
    
    for (feat_idx, feat_name) in enumerate(SHAP_FEATURE_NAMES)
        importance_values = [importance_by_tz[tz][feat_idx] for tz in valid_tz_shap]
        
        plot!(p3, valid_tz_shap, importance_values,
              linewidth=2.5,
              marker=SHAP_FEATURE_MARKERS[feat_idx],
              markersize=7,
              markeralpha=0.7,
              markerstrokewidth=0,
              color=SHAP_FEATURE_COLORS[feat_idx],
              linestyle=SHAP_FEATURE_LINESTYLES[feat_idx],
              label=feat_name,
              xlabel="Timezone Shift (hours)",
              ylabel=(ADAPTATION_SHAP_TARGET == "logdays" ? "Mean |SHAP| (log days)" : "Mean |SHAP| (days)"),
              title="",
              guidefont=("Times", 18, :bold),
              tickfont=("Times", 12),
              legend=(0.7, 0.95),
              legend_column=2,
              legendfont=("Times", 13),
              xlim=(0, 24),
              xticks=(4:4:24),
              ylim=(0, 1.5),
              grid=true,
              gridwidth=1.2,
              gridalpha=0.2,
              gridcolor=:gray60,
              gridstyle=:dash,
              linealpha=0.85)
    end
    
    # Add vertical line at 12 hours (transition point)
    vline!(p3, [12],
           linestyle=:dash,
           linewidth=1.5,
           color=:gray40,
           alpha=0.6,
           label="")
    
    # Combine all three plots
    p = plot(p1, p2, p3,
             layout=@layout([a; b; c]),
             size=(1400, 1140),
             dpi=300,
             link=:x)  # Link x-axes for better alignment
else
    # If no SHAP data, just use the two-panel plot
    p = plot(p1, p2,
             layout=@layout([a; b]),
             size=(1400, 760),
             dpi=300,
             link=:x)
end

# Save plot
plot_file = joinpath(OUTPUT_DIR, "phase_delay_proportion.png")
savefig(p, plot_file)

println("   ✓ Saved plot: $(basename(plot_file))")
println()

println("✅ Analysis complete!")


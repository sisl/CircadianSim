"""
Detect steady state in new timezone for jet lag trajectories
Lara Weed

This script detects when each CBTmin trajectory reaches steady state
in the new timezone (after the shift) and creates plots showing:
1. First day of steady state (median with percentiles)
2. Comparison of expected vs actual new steady state CBTmin
3. Difference between actual and expected CBTmin

WORKFLOW:
1. Load baseline steady state CBTmin from regular schedule (BASELINE_PST_DAYS)
2. Load CBTmin trajectories for all timezone shifts (1-23 hours)
3. For each particle:
   a. Use baseline steady state CBTmin from regular schedule (matched by particle index)
   b. Find first day in new timezone when steady state is reached
   c. Verify steady state CBTmin matches expected (target + shift) within ±15 min; the target is the settled
      pre-shift CBTmin (settled_preshift_cbtmin, revision_baseline.jl), the day-70 value where none exists
   Post-shift window per condition (INTERVENTION_TARGET_DAYS); included particles without a steady state in the
   window are saved as not adapted (right-censored at the window length).
   Only particles in included_particle_ids() (shared revision set) are analysed; each result carries its particle id.
   d. Determine adaptation direction (forward or backward)
4. Calculate statistics and create publication-ready plots
5. Save all results to JLD2 file
"""

# ================================================================
# PACKAGE SETUP
# ================================================================
using JLD2      # For loading/saving .jld2 files
using Plots     # For creating publication-ready plots
using Statistics # For mean, median, quantile, std
using StatsBase  # For quantile function
using CircadianPOMDP  # For get_CBTmin function

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

# ================================================================
# CONFIGURATION (from jetlag_reference.jl)
# ================================================================
# Condition: HighLight10000lux (default) or LowLight150lux. Pass via ARGS: julia detectSteadyState.jl LowLight150lux

const CONDITION = length(ARGS) >= 1 ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT

const BASELINE_DAYS = BASELINE_PST_DAYS
const SHIFT_DAY = BASELINE_PST_DAYS + 1
const TIMEZONE_SHIFTS = TIMEZONE_SHIFTS_FULL
const PARTICLE_HIST_DIR = jetlag_analysis_particle_hist_dir(condition=CONDITION)
const OUT_DIR = jetlag_analysis_dir(condition=CONDITION)
# STEADY_STATE_THRESHOLD, MIN_CONSECUTIVE_DAYS, EXPECTED_CBTMIN_TOLERANCE from jetlag_reference.jl

# ================================================================
# HELPER FUNCTIONS
# ================================================================
# unwrap24, wrap24, is_steady_state_window, find_baseline_steady_state_cbtmin,
# find_first_steady_state_day are now in the CircadianPOMDP package
# (src/analysis/steady_state.jl). Import via `using CircadianPOMDP`.

"""Total simulation days for jet lag (baseline + post-shift window of this condition)."""
const JETLAG_TOTAL_DAYS = jetlag_total_days(CONDITION)
const POST_SHIFT_WINDOW_DAYS = intervention_target_days(CONDITION)

# particle_hist_filename_for_shift is defined in jetlag_reference.jl

# events_to_trajectories is now imported from CircadianPOMDP

# ================================================================
# DATA LOADING
# ================================================================

println("📊 Loading baseline steady state CBTmin from regular schedule (condition: $CONDITION)...")

# Load events-based format (from simulate_particle_population_events_only)
regular_histories_path = regular_histories_file(condition=CONDITION)

if !isfile(regular_histories_path)
    error("Regular schedule particle histories not found at: $(regular_histories_path)")
end

baseline_steady_state_cbtmins, baseline_cbtmin_counts = load_baseline_steady_state_cbtmin(
    regular_histories_path; baseline_hours=BASELINE_DAYS * 24,
    min_consecutive_days=MIN_CONSECUTIVE_DAYS, threshold=Float64(STEADY_STATE_THRESHOLD))
num_particles = length(baseline_steady_state_cbtmins)
println("   ✓ Loaded baseline steady-state CBTmin for $num_particles particles")

n_with_baseline = count(x -> x !== nothing, baseline_steady_state_cbtmins)
println("   ✓ Extracted baseline steady state CBTmin for $(n_with_baseline) out of $(length(baseline_steady_state_cbtmins)) particles")
println("   Baseline CBTmin counts: min=$(minimum(baseline_cbtmin_counts)), max=$(maximum(baseline_cbtmin_counts)), mean=$(round(mean(baseline_cbtmin_counts), digits=1))")
# Target reference: settled pre-shift phase (slow settlers are not yet entrained at day 70)
settled_cbtmins = settled_preshift_cbtmin(CONDITION)
target_cbtmins = [s === nothing ? b : s for (s, b) in zip(settled_cbtmins, baseline_steady_state_cbtmins)]
println("   Target from the settled pre-shift phase ($(SETTLED_REFERENCE_DAYS) d); post-shift window $(POST_SHIFT_WINDOW_DAYS) d")
println("   Day 0 = last baseline CBTmin event (index = baseline_count for each particle)")
println()

println("📊 Loading CBTmin trajectories from particle histories...")

cbtmins_all = Vector{Vector{Vector{Float64}}}()

for tz_shift in TIMEZONE_SHIFTS
    filename = particle_hist_filename_for_shift(tz_shift; condition=CONDITION, total_days=JETLAG_TOTAL_DAYS)

    if !isfile(filename)
        println("   ⚠️  Timezone shift $(tz_shift) h: File not found, skipping")
        push!(cbtmins_all, Vector{Vector{Float64}}())
        continue
    end

    try
        local hist = load_particle_histories(filename)
        cbtmins = events_to_trajectories(hist.cbtmin_events, hist.num_particles)
        push!(cbtmins_all, cbtmins)
        println("   ✓ Timezone shift $(tz_shift) h: Loaded $(hist.num_particles) trajectories")
    catch e
        println("   ❌ Error loading timezone shift $(tz_shift) h: $e")
        push!(cbtmins_all, Vector{Vector{Float64}}())
    end
end

println("✅ Data loading complete!")
println()

# ================================================================
# STEADY STATE DETECTION
# ================================================================

println("🔍 Detecting steady state in new timezone...")
println("   Criteria:")
println("     - Range ≤ $(STEADY_STATE_THRESHOLD) hours for $(MIN_CONSECUTIVE_DAYS)+ consecutive days")
println("     - CBTmin within ±$(EXPECTED_CBTMIN_TOLERANCE) hours of expected (baseline + timezone shift)")
println("   Day 0 = last baseline CBTmin event from regular schedule (index = baseline_count per particle)")
println("   Day 1 = first CBTmin event in new timezone (index = baseline_count + 1)")
println("   Starting from first event in new timezone (day 1 relative to shift)")
println()

included_set = Set(included_particle_ids())   # shared particle set (revision)
println("   Analysing $(length(included_set)) included particles (shared revision set)")
steady_state_days_all = Vector{Vector{Int}}()  # Store steady state days for each timezone
steady_state_ids_all = Vector{Vector{Int}}()   # particle id for each entry of steady_state_days_all
not_adapted_ids_all = Vector{Vector{Int}}()    # included particles with no steady state within the window
baseline_cbtmins_all = Vector{Vector{Float64}}()  # Store baseline steady state CBTmin for each timezone
new_cbtmins_all = Vector{Vector{Float64}}()  # Store new steady state CBTmin for each timezone
adaptation_directions_all = Vector{Vector{Symbol}}()  # Store adaptation direction (:forward or :backward) for each particle

for (tz_idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    if tz_idx > length(cbtmins_all) || isempty(cbtmins_all[tz_idx])
        push!(steady_state_days_all, Int[])
        push!(steady_state_ids_all, Int[])
        push!(not_adapted_ids_all, Int[])
        push!(baseline_cbtmins_all, Float64[])
        push!(new_cbtmins_all, Float64[])
        push!(adaptation_directions_all, Symbol[])
        continue
    end
    
    cbtmins = cbtmins_all[tz_idx]
    steady_state_days = Int[]
    steady_state_ids = Int[]
    not_adapted_ids = Int[]
    baseline_cbtmins = Float64[]
    new_cbtmins = Float64[]
    adaptation_directions = Symbol[]
    
    # Process each particle, matching by index with baseline steady state values
    for (particle_idx, trajectory) in enumerate(cbtmins)
        particle_idx in included_set || continue
        if isempty(trajectory) || length(trajectory) < SHIFT_DAY + MIN_CONSECUTIVE_DAYS
            continue
        end
        
        # Get baseline steady state CBTmin from regular schedule (matched by particle index)
        if particle_idx > length(baseline_steady_state_cbtmins) || particle_idx > length(baseline_cbtmin_counts)
            continue  # Skip if particle index is out of range
        end
        
        baseline_count = baseline_cbtmin_counts[particle_idx]
        if baseline_steady_state_cbtmins[particle_idx] === nothing || baseline_count <= 0
            continue  # Skip if no baseline steady state found in regular schedule
        end
        baseline_cbtmin = target_cbtmins[particle_idx]

        # Find new timezone steady state and determine adaptation direction
        # Pass baseline_count so day 0 = last baseline CBTmin event
        first_steady_day, new_cbtmin, direction = find_first_steady_state_day(trajectory, baseline_cbtmin, tz_shift, baseline_count)
        if first_steady_day === nothing || new_cbtmin === nothing || direction === nothing
            push!(not_adapted_ids, particle_idx)   # not adapted within the window (censored)
            continue
        end
        
        push!(steady_state_days, first_steady_day)
        push!(steady_state_ids, particle_idx)
        push!(baseline_cbtmins, baseline_cbtmin)
        push!(new_cbtmins, new_cbtmin)
        push!(adaptation_directions, direction)
    end
    
    push!(steady_state_days_all, steady_state_days)
    push!(steady_state_ids_all, steady_state_ids)
    push!(not_adapted_ids_all, not_adapted_ids)
    push!(baseline_cbtmins_all, baseline_cbtmins)
    push!(new_cbtmins_all, new_cbtmins)
    push!(adaptation_directions_all, adaptation_directions)
    
    if !isempty(steady_state_days)
        println("   Timezone shift $(tz_shift) hours:")
        println("     $(length(steady_state_days)) trajectories reached steady state; $(length(not_adapted_ids)) not within $(POST_SHIFT_WINDOW_DAYS) d")
        println("     Median: $(round(median(steady_state_days), digits=1)) days")
        println("     Q25: $(round(quantile(steady_state_days, 0.25), digits=1)) days")
        println("     Q75: $(round(quantile(steady_state_days, 0.75), digits=1)) days")
    else
        println("   Timezone shift $(tz_shift) hours: No trajectories reached steady state")
    end
end

println()
println("✅ Steady state detection complete!")
println()

# ================================================================
# STATISTICS CALCULATION
# ================================================================

println("📊 Calculating statistics for plotting...")

medians = Float64[]
q25s = Float64[]
q75s = Float64[]
valid_timezones = Int[]

for (tz_idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    steady_days = steady_state_days_all[tz_idx]
    
    if !isempty(steady_days)
        push!(valid_timezones, tz_shift)
        push!(medians, median(steady_days))
        push!(q25s, quantile(steady_days, 0.25))
        push!(q75s, quantile(steady_days, 0.75))
    end
end

println("   ✓ Calculated statistics for $(length(valid_timezones)) timezone shifts")
println()

# ================================================================
# PLOTTING
# ================================================================

println("📈 Creating steady state detection plot...")

# Set publication-ready defaults
Plots.default(fontfamily="Times", dpi=300, linewidth=2.5)

# Ensure output directory exists (use reference for subdir name)
mkpath(OUT_DIR)
STEADY_STATE_OUTPUT_DIR = joinpath(OUT_DIR, SUBDIR_STEADY_STATE)
mkpath(STEADY_STATE_OUTPUT_DIR)

# Create plot with shaded percentiles
p = plot(size=(1200, 600),
         background_color=:white, foreground_color=:black,
         framestyle=:box,
         margin=10Plots.mm, bottom_margin=12Plots.mm, left_margin=12Plots.mm,
         top_margin=8Plots.mm, right_margin=8Plots.mm)

# Plot shaded region for 25th-75th percentiles
plot!(p, valid_timezones, q75s, 
      fillrange=q25s, 
      fillalpha=0.3, 
      fillcolor=:lightblue,
      linealpha=0.0,
      label="")

# Plot median line
plot!(p, valid_timezones, medians,
      linewidth=3,
      color=:black,
      marker=:circle,
      markersize=6,
      label="Median",
      title="First Day of Steady State in New Timezone",
      xlabel="Timezone Shift (hours)",
      ylabel="Days Relative to Shift (0 = last day in old timezone)",
      titlefont=("Times", 20, :bold), titlefontsize=20,
      guidefont=("Times", 16), guidefontsize=16,
      tickfont=("Times", 14), tickfontsize=14,
      grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
      legend=:topright, legendfont=("Times", 14), legendfontsize=14)

# Add reference line at y=0 (shift day)
plot!(p, [minimum(valid_timezones), maximum(valid_timezones)], [0, 0],
      linestyle=:dash, linewidth=2, color=:gray50, alpha=0.5, label="Shift day")

# Save plot
out_file = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_detection.png")
out_file_pdf = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_detection.pdf")
savefig(p, out_file)
#savefig(p, out_file_pdf)

println("✅ Plot saved:")
println("   PNG: $(out_file)")
#println("   PDF: $(out_file_pdf)")
println()

# ================================================================
# COMPARE BASELINE VS NEW STEADY STATE CBTMIN
# ================================================================

println("🔍 Comparing baseline vs new steady state CBTmin values...")
println("   Expected new CBTmin = baseline CBTmin + timezone shift (wrapped to 0-24)")
println()

# Calculate expected and actual CBTmin values for each timezone
expected_cbtmins_all = Vector{Vector{Float64}}()
actual_cbtmins_all = Vector{Vector{Float64}}()
differences_all = Vector{Vector{Float64}}()
valid_tz_for_comparison = Int[]

for (tz_idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    baseline_cbtmins = baseline_cbtmins_all[tz_idx]
    new_cbtmins = new_cbtmins_all[tz_idx]
    directions = adaptation_directions_all[tz_idx]
    
    if isempty(baseline_cbtmins) || isempty(new_cbtmins) || isempty(directions)
        continue
    end
    
    expected_cbtmins = Float64[]
    actual_cbtmins = Float64[]
    differences = Float64[]
    
    for i in eachindex(baseline_cbtmins)
        baseline = baseline_cbtmins[i]
        actual = new_cbtmins[i]
        direction = directions[i]
        
        # Expected value is always baseline + shift (wrapped), regardless of adaptation direction.
        # The direction only tells us HOW the particle adapted (forward or backward path),
        # but the destination should always be the same: baseline + tz_shift in the new timezone.
        # For shifts > 12 hours, both forward and backward paths should converge to the same
        # wrapped value, but we use the forward calculation as the ground truth.
        expected = wrap24(baseline + tz_shift)
        
        # Calculate difference (accounting for wrapping)
        # We want to find the shortest angular distance around the 24-hour circle
        diff = actual - expected
        # Wrap difference to -12 to +12 range for interpretation
        # This ensures we measure the shortest path (e.g., if expected=23 and actual=1, diff=-2 not +22)
        if diff > 12
            diff -= 24
        elseif diff < -12
            diff += 24
        end
        
        push!(expected_cbtmins, expected)
        push!(actual_cbtmins, actual)
        push!(differences, diff)
    end
    
    push!(expected_cbtmins_all, expected_cbtmins)
    push!(actual_cbtmins_all, actual_cbtmins)
    push!(differences_all, differences)
    push!(valid_tz_for_comparison, tz_shift)
    
    # Print summary statistics
    mean_diff = mean(differences)
    std_diff = std(differences)
    median_diff = median(differences)
    
    # Count adaptation directions
    n_forward = count(d -> d == :forward, directions)
    n_backward = count(d -> d == :backward, directions)
    
    println("   Timezone shift $(tz_shift) hours:")
    println("     Adaptation directions: $(n_forward) forward, $(n_backward) backward")
    println("     Mean difference (actual - expected): $(round(mean_diff, digits=2)) ± $(round(std_diff, digits=2)) hours")
    println("     Median difference: $(round(median_diff, digits=2)) hours")
    println("     Mean baseline CBTmin: $(round(mean(baseline_cbtmins), digits=2)) hours")
    println("     Mean expected new CBTmin: $(round(mean(expected_cbtmins), digits=2)) hours")
    println("     Mean actual new CBTmin: $(round(mean(actual_cbtmins), digits=2)) hours")
    
    # For large shifts (>15 hours), print detailed diagnostics
    if tz_shift > 15 && length(baseline_cbtmins) > 0
        println("     [$(tz_shift)h shift diagnostics:]")
        n_samples = min(10, length(baseline_cbtmins))
        println("       Sample baseline values: $(round.(baseline_cbtmins[1:n_samples], digits=2))")
        println("       Sample directions: $(directions[1:n_samples])")
        println("       Sample expected values: $(round.(expected_cbtmins[1:n_samples], digits=2))")
        println("       Sample actual values: $(round.(actual_cbtmins[1:n_samples], digits=2))")
        println("       Sample differences: $(round.(differences[1:n_samples], digits=2))")
        
        # Show examples where difference is large
        large_diff_indices = findall(d -> abs(d) > 2.0, differences)
        if !isempty(large_diff_indices)
            println("       [Large differences (>2h) found in $(length(large_diff_indices)) particles:]")
            for idx in large_diff_indices[1:min(5, length(large_diff_indices))]
                println("         Particle $(idx): baseline=$(round(baseline_cbtmins[idx], digits=2)), " *
                       "expected=$(round(expected_cbtmins[idx], digits=2)), " *
                       "actual=$(round(actual_cbtmins[idx], digits=2)), " *
                       "diff=$(round(differences[idx], digits=2)), " *
                       "direction=$(directions[idx])")
            end
        end
    end
end

println()
println("✅ Comparison complete!")
println()

# ================================================================
# PLOT COMPARISON
# ================================================================

println("📈 Creating baseline vs new steady state CBTmin comparison plot...")

# Calculate statistics for plotting
expected_medians = Float64[]
expected_q25s = Float64[]
expected_q75s = Float64[]
actual_medians = Float64[]
actual_q25s = Float64[]
actual_q75s = Float64[]
diff_medians = Float64[]
diff_q25s = Float64[]
diff_q75s = Float64[]

for (tz_idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    if !(tz_shift in valid_tz_for_comparison)
        continue
    end
    
    idx = findfirst(x -> x == tz_shift, valid_tz_for_comparison)
    if idx === nothing
        continue
    end
    
    expected = expected_cbtmins_all[idx]
    actual = actual_cbtmins_all[idx]
    diffs = differences_all[idx]
    
    push!(expected_medians, median(expected))
    push!(expected_q25s, quantile(expected, 0.25))
    push!(expected_q75s, quantile(expected, 0.75))
    
    push!(actual_medians, median(actual))
    push!(actual_q25s, quantile(actual, 0.25))
    push!(actual_q75s, quantile(actual, 0.75))
    
    push!(diff_medians, median(diffs))
    push!(diff_q25s, quantile(diffs, 0.25))
    push!(diff_q75s, quantile(diffs, 0.75))
end

# Create comparison plot
p_compare = plot(size=(1400, 600),
                 background_color=:white, foreground_color=:black,
                 framestyle=:box,
                 margin=10Plots.mm, bottom_margin=12Plots.mm, left_margin=12Plots.mm,
                 top_margin=8Plots.mm, right_margin=8Plots.mm)

# Plot expected (baseline + shift) with shading
plot!(p_compare, valid_tz_for_comparison, expected_q75s,
      fillrange=expected_q25s,
      fillalpha=0.2,
      fillcolor=:lightblue,
      linealpha=0.0,
      label="")

plot!(p_compare, valid_tz_for_comparison, expected_medians,
      linewidth=2.5,
      linestyle=:dash,
      color=:blue,
      marker=:circle,
      markersize=5,
      label="Expected (baseline + shift)",
      title="Baseline vs New Steady State CBTmin",
      xlabel="Timezone Shift (hours)",
      ylabel="CBTmin (hours)",
      titlefont=("Times", 20, :bold), titlefontsize=20,
      guidefont=("Times", 16), guidefontsize=16,
      tickfont=("Times", 14), tickfontsize=14,
      grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
      legend=:topright, legendfont=("Times", 14), legendfontsize=14,
      ylim=(0, 24))

# Plot actual (new steady state) with shading
plot!(p_compare, valid_tz_for_comparison, actual_q75s,
      fillrange=actual_q25s,
      fillalpha=0.2,
      fillcolor=:lightcoral,
      linealpha=0.0,
      label="")

plot!(p_compare, valid_tz_for_comparison, actual_medians,
      linewidth=3,
      linestyle=:solid,
      color=:red,
      marker=:circle,
      markersize=6,
      label="Actual (new steady state)")

# Save comparison plot
compare_file = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_cbtmin_comparison.png")
compare_file_pdf = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_cbtmin_comparison.pdf")
savefig(p_compare, compare_file)
savefig(p_compare, compare_file_pdf)

println("✅ Comparison plot saved:")
println("   PNG: $(compare_file)")
#println("   PDF: $(compare_file_pdf)")
println()

# Create difference plot
p_diff = plot(size=(1400, 600),
              background_color=:white, foreground_color=:black,
              framestyle=:box,
              margin=10Plots.mm, bottom_margin=12Plots.mm, left_margin=12Plots.mm,
              top_margin=8Plots.mm, right_margin=8Plots.mm)

# Plot difference with shading
plot!(p_diff, valid_tz_for_comparison, diff_q75s,
      fillrange=diff_q25s,
      fillalpha=0.3,
      fillcolor=:lightgray,
      linealpha=0.0,
      label="")

plot!(p_diff, valid_tz_for_comparison, diff_medians,
      linewidth=3,
      color=:black,
      marker=:circle,
      markersize=6,
      label="Median difference",
      title="Difference: Actual - Expected CBTmin",
      xlabel="Timezone Shift (hours)",
      ylabel="Difference (hours)",
      titlefont=("Times", 20, :bold), titlefontsize=20,
      guidefont=("Times", 16), guidefontsize=16,
      tickfont=("Times", 14), tickfontsize=14,
      grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
      legend=:topright, legendfont=("Times", 14), legendfontsize=14)

# Add reference line at y=0
plot!(p_diff, [minimum(valid_tz_for_comparison), maximum(valid_tz_for_comparison)], [0, 0],
      linestyle=:dash, linewidth=2, color=:gray50, alpha=0.5, label="Perfect match")

# Save difference plot
diff_file = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_cbtmin_difference.png")
diff_file_pdf = joinpath(STEADY_STATE_OUTPUT_DIR, "steady_state_cbtmin_difference.pdf")
savefig(p_diff, diff_file)
#savefig(p_diff, diff_file_pdf)

println("✅ Difference plot saved:")
println("   PNG: $(diff_file)")
#println("   PDF: $(diff_file_pdf)")
println()

# ================================================================
# SAVE RESULTS
# ================================================================

println("💾 Saving steady state detection results...")

results_data = Dict(
    "steady_state_days_by_timezone" => steady_state_days_all,
    "steady_state_particle_ids_by_timezone" => steady_state_ids_all,
    "not_adapted_particle_ids_by_timezone" => not_adapted_ids_all,   # censored at post_shift_window_days
    "post_shift_window_days" => POST_SHIFT_WINDOW_DAYS,
    "target_reference" => "settled pre-shift CBTmin after $(SETTLED_REFERENCE_DAYS) d ($(TARGET_REFERENCE_TAG))",
    "included_particle_ids" => sort(collect(included_set)),
    "baseline_cbtmins_by_timezone" => baseline_cbtmins_all,
    "new_cbtmins_by_timezone" => new_cbtmins_all,
    "adaptation_directions_by_timezone" => adaptation_directions_all,
    "expected_cbtmins_by_timezone" => expected_cbtmins_all,
    "actual_cbtmins_by_timezone" => actual_cbtmins_all,
    "differences_by_timezone" => differences_all,
    "timezone_shifts" => TIMEZONE_SHIFTS,
    "statistics" => Dict(
        "medians" => medians,
        "q25s" => q25s,
        "q75s" => q75s,
        "valid_timezones" => valid_timezones
    ),
    "comparison_statistics" => Dict(
        "expected_medians" => expected_medians,
        "expected_q25s" => expected_q25s,
        "expected_q75s" => expected_q75s,
        "actual_medians" => actual_medians,
        "actual_q25s" => actual_q25s,
        "actual_q75s" => actual_q75s,
        "diff_medians" => diff_medians,
        "diff_q25s" => diff_q25s,
        "diff_q75s" => diff_q75s,
        "valid_timezones_for_comparison" => valid_tz_for_comparison
    ),
    "detection_parameters" => Dict(
        "steady_state_threshold" => STEADY_STATE_THRESHOLD,
        "min_consecutive_days" => MIN_CONSECUTIVE_DAYS,
        "shift_day" => SHIFT_DAY,
        "baseline_days" => BASELINE_DAYS
    )
)

results_file = steady_state_results_path(condition=CONDITION)
save(results_file, results_data)

println("✅ Results saved to: $(results_file)")
println()
println("✅ Steady state detection analysis complete!")


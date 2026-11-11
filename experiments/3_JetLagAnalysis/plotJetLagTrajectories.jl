"""
Plot CBTmin adaptation trajectories from day -7 to +TARGET_DAYS around timezone transitions
Lara Weed

Fixed window: 7 days before shift to TARGET_DAYS_FULL days after shift.
"""

# ================================================================
# PACKAGE SETUP
# ================================================================
using CircadianPOMDP
using JLD2
using Plots
using Statistics
using StatsBase

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

# ================================================================
# CONFIGURATION (from jetlag_reference.jl)
# ================================================================
# Condition: HighLight10000lux (default) or LowLight150lux. Pass via ARGS: julia plotJetLagTrajectories.jl LowLight150lux

const CONDITION = length(ARGS) >= 1 ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT

const BASELINE_DAYS = BASELINE_PST_DAYS
const SHIFT_DAY = BASELINE_PST_DAYS + 1
const TIMEZONE_SHIFTS = TIMEZONE_SHIFTS_FULL
const JETLAG_TOTAL_DAYS = jetlag_total_days(CONDITION)   # the analysis histories (150 lux: 100-day window, plotted to day 50)
const PLOT_POST_SHIFT_DAYS = TARGET_DAYS_FULL  # plot window: day -7 to +PLOT_POST_SHIFT_DAYS
const PLOT_ALL_PARTICLES = true
const PLOT_ONLY_STABILIZED = true
const PARTICLE_HIST_DIR = jetlag_analysis_particle_hist_dir(condition=CONDITION)
const OUT_DIR = jetlag_analysis_trajectories_dir(condition=CONDITION)

# ================================================================
# HELPER FUNCTIONS
# ================================================================
# unwrap24, wrap24, is_steady_state_window are now in the CircadianPOMDP package.

"""
Detect if a particle reached steady state by the end of the baseline period.
Uses `find_baseline_steady_state_cbtmin` from CircadianPOMDP to check
whether a steady window exists within the baseline trajectory.
"""
function reached_steady_state_by_end(cbtmin_trajectory, baseline_days=BASELINE_DAYS, min_consecutive_days=7, threshold=0.25)
    n = length(cbtmin_trajectory)
    n < baseline_days && return false
    baseline = cbtmin_trajectory[1:min(n, baseline_days)]
    return find_baseline_steady_state_cbtmin(baseline; min_consecutive_days, threshold) !== nothing
end

# particle_hist_filename_for_shift is defined in jetlag_reference.jl

# events_to_trajectories is now imported from CircadianPOMDP

# ================================================================
# DATA LOADING (from particle histories: events → trajectories)
# ================================================================

println("📊 Loading particle histories from $(PARTICLE_HIST_DIR) (condition: $CONDITION)...")

cbtmins_all = Vector{Vector{Vector{Float64}}}()
sleep_onsets_all = Vector{Vector{Vector{Float64}}}()
sleep_offsets_all = Vector{Vector{Vector{Float64}}}()

for tz_shift in TIMEZONE_SHIFTS
    filename = particle_hist_filename_for_shift(tz_shift; condition=CONDITION, total_days=JETLAG_TOTAL_DAYS)
    println("   Timezone shift $(tz_shift) h: $(basename(filename))")

    if !isfile(filename)
        println("   ⚠️  File not found; skipping.")
        push!(cbtmins_all, Vector{Vector{Float64}}())
        push!(sleep_onsets_all, Vector{Vector{Float64}}())
        push!(sleep_offsets_all, Vector{Vector{Float64}}())
        continue
    end

    try
        hist = load_particle_histories(filename)
        cbtmins = events_to_trajectories(hist.cbtmin_events, hist.num_particles)
        sleep_onsets = events_to_trajectories(hist.sleep_onset_events, hist.num_particles)
        sleep_offsets = events_to_trajectories(hist.sleep_offset_events, hist.num_particles)

        push!(cbtmins_all, cbtmins)
        push!(sleep_onsets_all, sleep_onsets)
        push!(sleep_offsets_all, sleep_offsets)
        println("   ✓ $(hist.num_particles) particles, CBTmin/sleep onset/sleep offset trajectories")
    catch e
        println("   ❌ Error: $e")
        push!(cbtmins_all, Vector{Vector{Float64}}())
        push!(sleep_onsets_all, Vector{Vector{Float64}}())
        push!(sleep_offsets_all, Vector{Vector{Float64}}())
    end
end

println("✅ Data loading complete!")
println()

# ================================================================
# ALIGNMENT: COUNT BASELINE CBTMIN EVENTS
# ================================================================

println("🔍 Counting baseline CBTmin events from regular schedule ($(BASELINE_DAYS) days)...")

# Load events-based format (from simulate_particle_population_events_only)
regular_histories_path = regular_histories_file(condition=CONDITION)

if !isfile(regular_histories_path)
    error("Regular schedule particle histories not found at: $(regular_histories_path)")
end

hist = load_particle_histories(regular_histories_path)
baseline_hours = BASELINE_DAYS * 24
baseline_cbtmin_trajectories, baseline_cbtmin_counts = baseline_cbtmin_from_events(
    hist.cbtmin_events, hist.num_particles, baseline_hours)
println("   ✓ Loaded events for $(hist.num_particles) particles")

println("   ✓ Counted baseline CBTmin events for $(length(baseline_cbtmin_counts)) particles")
println("   Baseline CBTmin counts: min=$(minimum(baseline_cbtmin_counts)), max=$(maximum(baseline_cbtmin_counts)), mean=$(round(mean(baseline_cbtmin_counts), digits=1))")

# Detect which particles reached steady state by the end of baseline
println("🔍 Detecting particles that stabilized by end of baseline ($(BASELINE_DAYS) days)...")
stabilized_particles = BitVector()

for (i, cbtmin_traj) in enumerate(baseline_cbtmin_trajectories)
    if isempty(cbtmin_traj)
        push!(stabilized_particles, false)
    else
        # baseline_cbtmin_trajectories already contains only baseline-period events
        baseline_count = baseline_cbtmin_counts[i]
        if baseline_count > 0 && baseline_count <= length(cbtmin_traj)
            baseline_traj = cbtmin_traj[1:baseline_count]
            # Check if there's a steady state window in the last part of baseline
            # We need at least 7 consecutive days, so check if last 7+ events are stable
            stabilized = reached_steady_state_by_end(baseline_traj, baseline_count, 7, 0.25)
            push!(stabilized_particles, stabilized)
        else
            push!(stabilized_particles, false)
        end
    end
end

# Shared particle set (revision): plot only included particles
included_set = Set(included_particle_ids())
stabilized_particles .&= [i in included_set for i in eachindex(stabilized_particles)]

n_stabilized = count(stabilized_particles)
n_total = length(stabilized_particles)
println("   ✓ Detected steady state for $(n_stabilized) out of $(n_total) particles ($(round(100*n_stabilized/n_total, digits=1))%)")

# Verify particle count matches jet lag trajectories
if !isempty(cbtmins_all) && !isempty(cbtmins_all[1])
    jetlag_particle_count = length(cbtmins_all[1])
    if length(baseline_cbtmin_counts) != jetlag_particle_count
        @warn "Particle count mismatch: baseline=$(length(baseline_cbtmin_counts)), jetlag=$(jetlag_particle_count)"
        @warn "This may indicate particles are not matched correctly!"
    else
        println("   ✓ Verified particle count matches jet lag trajectories ($(jetlag_particle_count) particles)")
    end
end
println()

# ================================================================
# PLOTTING: TRAJECTORIES FROM DAY -7 TO +TARGET_DAYS
# ================================================================

println("📈 Plotting CBTmin trajectories from day -7 to +$(PLOT_POST_SHIFT_DAYS) around timezone transitions...")

# Ensure output directory exists (Data/JetLag/3_JetLagAnalysis/trajectories)
mkpath(OUT_DIR)

# Set high DPI and Times font for publication quality (matching compareWithNHANES.jl style)
Plots.default(fontfamily="Times", dpi=300, linewidth=2.5)

# Shared trajectory style for CBTmin and sleep plots
trajectory_color = :black
trajectory_alpha = 0.05

# 5x5 grid for up to 23 timezone shifts
# Publication-ready settings: minimized margins to maximize plot area, larger fonts
# - Times serif font family
# - White background for print compatibility
# - Reduced margins to maximize plot area relative to labels
# CBTmin figure (S9/S10) in the manuscript style (figure_style.jl): 6 × 4 panels drawn for 6.5 × 7.0 in (SI text width,
# full page with caption room), shift label inside each panel, y axis on the left column and x axis on the bottom panels only,
# no title
const TRAJ_NROW, TRAJ_NCOL = 6, 4
const TRAJ_TICK_FS, TRAJ_GUIDE_FS, TRAJ_LABEL_FS = 24, 25, 27   # at 6.5 in wide: 9.8, 10.2, 11.0 pt
p_traj = plot(layout=(TRAJ_NROW, TRAJ_NCOL), legend=false, size=(1600, round(Int, 1600 * FULL_PAGE_H_IN / SI_WIDTH_IN)),
              margin=1Plots.mm, bottom_margin=2Plots.mm, left_margin=2Plots.mm,
              top_margin=1Plots.mm, right_margin=1Plots.mm,
              background_color=:white, foreground_color=:black,
              framestyle=:box)

for (idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    if idx > length(cbtmins_all)
        break
    end

    traces = cbtmins_all[idx]

    if isempty(traces)
        plot!(p_traj[idx], [], [], title = "$(tz_shift) h (No Data)",
              xlabel = "Days from shift", ylabel = "CBTmin (h)",
              titlefont=("Times", 18, :bold), titlefontsize=18,
              guidefont=("Times", 16), guidefontsize=16,
              tickfont=("Times", 14), tickfontsize=14,
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
              framestyle=:box)
        continue
    end

    # Filter to stabilized particles if requested
    available_indices = collect(1:length(traces))
    if PLOT_ONLY_STABILIZED
        available_indices = [i for i in available_indices if i <= length(stabilized_particles) && stabilized_particles[i]]
    end
    
    # Use all available particles (no subsampling)
    plot_indices = available_indices
    traces_to_plot = [traces[i] for i in plot_indices]
    if isempty(traces_to_plot)
        println("   ⚠️  No particles available to plot!")
    end

    # For each particle, focus on a fixed window from day -7 to +PLOT_POST_SHIFT_DAYS
    # relative to the shift (t = 0 at day 35, which is the last day in old timezone)
    #
    # Alignment: Use the actual number of CBTmin events during the baseline
    # from the regular schedule to determine when day 35 (t = 0) occurs in each particle's
    # jet lag trajectory. This accounts for particles with different circadian periods.
    #
    # Particle matching: Both regular schedule and jet lag simulations use the same
    # initial_particles loaded from the same file, and simulate_particle_from_initial_population
    # preserves particle order. Therefore, particle index i in jet lag trajectories corresponds
    # to particle index i in the regular schedule baseline.
    # Zero line (reviewer 2.7): solid, darker than the grid, under the trajectories (same style as Fig 3)
    plot!(p_traj[idx], [-7, PLOT_POST_SHIFT_DAYS], [0, 0], color=:black, alpha=0.85, lw=1.6, linestyle=:solid)
    for (j, trace_any) in enumerate(traces_to_plot)
        trace = collect(trace_any)  # ensure indexable vector
        n = length(trace)
        
        # Get the original particle index (before random sampling)
        # plot_indices[j] gives us the particle index in the jet lag simulation
        original_particle_idx = plot_indices[j]
        
        # Get the number of CBTmin events this particle had during baseline
        # Match by particle index: particle i in jet lag = particle i in baseline
        if original_particle_idx < 1 || original_particle_idx > length(baseline_cbtmin_counts)
            # Fallback if particle index is out of range (shouldn't happen if counts match)
            @warn "Particle index $(original_particle_idx) out of range for baseline counts (1-$(length(baseline_cbtmin_counts)))"
            baseline_count = BASELINE_DAYS
        else
            baseline_count = baseline_cbtmin_counts[original_particle_idx]
        end
        
        # Align: event index = baseline_count corresponds to day 35 (t = 0, last day in old timezone)
        # Event index = baseline_count + 1 corresponds to day 36 (t = 1, first day in new timezone)
        # So event index = baseline_count - 7 corresponds to t = -7
        # And event index = baseline_count + PLOT_POST_SHIFT_DAYS corresponds to t = +PLOT_POST_SHIFT_DAYS
        day_zero_idx = baseline_count  # Last day in old timezone
        start_idx = max(1, day_zero_idx - 7)  # t = -7
        end_idx   = min(n, day_zero_idx + PLOT_POST_SHIFT_DAYS)  # t = +PLOT_POST_SHIFT_DAYS
        
        if end_idx <= start_idx || day_zero_idx > n
            continue
        end

        window = trace[start_idx:end_idx]
        x_unwrapped = unwrap24_centered(window)

        # Time axis: event at day_zero_idx = t = 0 (day 35, last day in old timezone)
        # So we subtract day_zero_idx to center at the shift
        t = collect((start_idx:end_idx) .- day_zero_idx)

        # One line per particle (unwrap24 gives a continuous trajectory in [-12,12]-style coordinates)
        plot!(p_traj[idx], t, x_unwrapped, alpha=trajectory_alpha, lw=0.8, color=trajectory_color,
              title="", ylim=(-24, 24), xlim=(-7, PLOT_POST_SHIFT_DAYS),
              titlefont=("Times", 18, :bold), titlefontsize=18,
              guidefont=("Times", 16), guidefontsize=16,
              tickfont=("Times", 14), tickfontsize=14,
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
              framestyle=:box)
    end

    # Reference lines: shift at t = 0 (last day in old timezone), and CBTmin at ±24 hours (0 is drawn above)
    # Make reference lines more prominent for publication
    plot!(p_traj[idx], [0, 0], [-24, 24], alpha=0.7, lw=2, color=:black,
          linestyle=:dash, xlim=(-7, PLOT_POST_SHIFT_DAYS), linewidth=2)
    plot!(p_traj[idx], [-7, PLOT_POST_SHIFT_DAYS], [24, 24], alpha=0.4, lw=1,
          linestyle=:dash, color=:gray50)
    plot!(p_traj[idx], [-7, PLOT_POST_SHIFT_DAYS], [-24, -24], alpha=0.4, lw=1,
          linestyle=:dash, color=:gray50)

    # Axes on the outer panels only; the last panel of each column carries the x axis
    left_col = mod1(idx, TRAJ_NCOL) == 1
    bottom = idx + TRAJ_NCOL > length(TIMEZONE_SHIFTS)
    last_row = idx > (TRAJ_NROW - 1) * TRAJ_NCOL
    plot!(p_traj[idx], xlabel=(last_row ? "Days" : ""), ylabel="",   # one shared y label, added at save
          yticks=-20:10:20, xformatter=(last_row ? :auto : (_ -> "")), yformatter=(left_col ? :auto : (_ -> "")),
          bottom_margin=(last_row ? 8Plots.mm : 0Plots.mm), left_margin=(left_col ? 3Plots.mm : 0Plots.mm), right_margin=(mod1(idx, TRAJ_NCOL) == TRAJ_NCOL ? 4Plots.mm : 0Plots.mm),
          guidefont=(FIG_BOLD, TRAJ_GUIDE_FS), tickfont=(FIG_BOLD, TRAJ_TICK_FS),
          xguidefont=(FIG_BOLD, TRAJ_GUIDE_FS), yguidefont=(FIG_BOLD, TRAJ_GUIDE_FS))
    # A bottom panel above an empty slot (20 h): x axis drawn as annotations into that slot, so its row keeps the
    # same height as the others (tick labels would make the layout reserve space for the whole row)
    if bottom && !last_row
        for x in 0:10:PLOT_POST_SHIFT_DAYS
            annotate!(p_traj[idx], x, -28.5, text(string(x), FIG_BOLD, TRAJ_TICK_FS))
        end
        # its "Days" label goes in the empty slot below (an annotation here would be clipped), see below
    end
    # Shift label: top right while the trajectories stay low (≤ 8 h), bottom right once they reach the top
    annotate!(p_traj[idx], PLOT_POST_SHIFT_DAYS - 1.5, tz_shift <= 8 ? 19.5 : -19.5,
              text("$(tz_shift)h", FIG_BOLD, TRAJ_LABEL_FS, :right))
end
for k in (length(TIMEZONE_SHIFTS) + 1):(TRAJ_NROW * TRAJ_NCOL)   # unused grid slots
    plot!(p_traj[k], framestyle=:none, grid=false, showaxis=false, ticks=false)
    # "Days" for the panel above (20 h): same horizontal margins so the plot areas line up, just below its tick labels
    if k - TRAJ_NCOL >= 1 && k - TRAJ_NCOL <= length(TIMEZONE_SHIFTS)
        plot!(p_traj[k], xlims=(-7, PLOT_POST_SHIFT_DAYS), ylims=(0, 1), left_margin=0Plots.mm,
              right_margin=(mod1(k, TRAJ_NCOL) == TRAJ_NCOL ? 4Plots.mm : 0Plots.mm))
        annotate!(p_traj[k], (PLOT_POST_SHIFT_DAYS - 7) / 2, 0.85, text("Days", FIG_BOLD, TRAJ_GUIDE_FS))
    end
end

# Save at high resolution for publication (300 DPI set via Plots.default above)
filename_suffix = PLOT_ONLY_STABILIZED ? "_stabilized_alpha005" : "_alpha005"
out_file = joinpath(OUT_DIR, "cbtmin_jetlag_trajectories$(filename_suffix).png")
# One y label for the whole grid, centred on the panel rows (a label per row is as tall as the row and crowds it)
ylabel_panel = plot(framestyle=:none, showaxis=false, grid=false, ticks=false, legend=false, xlims=(0, 1), ylims=(0, 1),
                    margin=0Plots.mm)
annotate!(ylabel_panel, 0.5, 0.54, text(raw"CBT_{MIN} (Hour)\ ", FIG_BOLD, TRAJ_GUIDE_FS, rotation=90))
traj_fig = plot(ylabel_panel, p_traj; layout=grid(1, 2, widths=[0.03, 0.97]),
                size=(1600, round(Int, 1600 * FULL_PAGE_H_IN / SI_WIDTH_IN)))
savefig(traj_fig, out_file)

# Also save a PDF version (vector format, preferred for publications)
#out_file_pdf = joinpath(OUT_DIR, "cbtmin_jetlag_trajectories_minus7_to_50_alpha005.pdf")
#savefig(p_traj, out_file_pdf)

println("✅ CBTmin trajectory plot (-7 to +$(PLOT_POST_SHIFT_DAYS)) saved:")
println("   PNG (300 DPI): $(out_file)")

# ================================================================
# SLEEP ONSET TRAJECTORIES (separate figure)
# ================================================================

println("📈 Plotting sleep onset trajectories (day -7 to +$(PLOT_POST_SHIFT_DAYS))...")

p_onset = plot(layout=(5, 5), legend=false, size=(2400, 2400),
              margin=5Plots.mm, bottom_margin=8Plots.mm, left_margin=10Plots.mm,
              top_margin=6Plots.mm, right_margin=5Plots.mm,
              background_color=:white, foreground_color=:black,
              framestyle=:box)

for (idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    idx > length(sleep_onsets_all) && break
    traces = sleep_onsets_all[idx]

    if isempty(traces)
        plot!(p_onset[idx], [], [], title = "$(tz_shift) h (No Data)",
              xlabel = "Days from shift", ylabel = "Sleep onset (h)",
              titlefont=("Times", 18, :bold), guidefont=("Times", 16), tickfont=("Times", 14),
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black, framestyle=:box)
        continue
    end

    available_indices = collect(1:length(traces))
    if PLOT_ONLY_STABILIZED
        available_indices = [i for i in available_indices if i <= length(stabilized_particles) && stabilized_particles[i]]
    end
    plot_indices = available_indices
    traces_to_plot = [traces[i] for i in plot_indices]

    for (j, trace_any) in enumerate(traces_to_plot)
        trace = collect(trace_any)
        n = length(trace)
        original_particle_idx = plot_indices[j]
        baseline_count = (original_particle_idx >= 1 && original_particle_idx <= length(baseline_cbtmin_counts)) ?
            baseline_cbtmin_counts[original_particle_idx] : BASELINE_DAYS
        day_zero_idx = baseline_count
        start_idx = max(1, day_zero_idx - 7)
        end_idx   = min(n, day_zero_idx + PLOT_POST_SHIFT_DAYS)
        (end_idx <= start_idx || day_zero_idx > n) && continue

        window = trace[start_idx:end_idx]
        x_unwrapped = unwrap24_centered(window)
        t = collect((start_idx:end_idx) .- day_zero_idx)

        plot!(p_onset[idx], t, x_unwrapped, alpha=trajectory_alpha, lw=0.8, color=trajectory_color,
              title="$(tz_shift) h", ylim=(-24, 24), xlim=(-7, PLOT_POST_SHIFT_DAYS),
              titlefont=("Times", 18, :bold), guidefont=("Times", 16), tickfont=("Times", 14),
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black, framestyle=:box)
    end

    plot!(p_onset[idx], [0, 0], [-24, 24], alpha=0.7, lw=2, color=:black, linestyle=:dash, xlim=(-7, PLOT_POST_SHIFT_DAYS), linewidth=2)
    plot!(p_onset[idx], [-7, PLOT_POST_SHIFT_DAYS], [0, 0], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_onset[idx], [-7, PLOT_POST_SHIFT_DAYS], [24, 24], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_onset[idx], [-7, PLOT_POST_SHIFT_DAYS], [-24, -24], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_onset[idx], xlabel="Days relative to shift", ylabel="Sleep onset (h)",
          guidefont=("Times", 16), tickfont=("Times", 14), xguidefont=("Times", 16), yguidefont=("Times", 16))
end

plot!(p_onset, plot_title="Sleep Onset Trajectories Around Timezone Shifts\n(Window: day -7 to +$(PLOT_POST_SHIFT_DAYS) relative to shift)",
      plot_titlefontsize=24, titlefont=("Times", 24, :bold))
out_onset = joinpath(OUT_DIR, "sleep_onset_jetlag_trajectories$(filename_suffix).png")
savefig(p_onset, out_onset)
println("   Saved: $(out_onset)")

# ================================================================
# SLEEP OFFSET TRAJECTORIES (separate figure)
# ================================================================

println("📈 Plotting sleep offset trajectories (day -7 to +$(PLOT_POST_SHIFT_DAYS))...")

p_offset = plot(layout=(5, 5), legend=false, size=(2400, 2400),
               margin=5Plots.mm, bottom_margin=8Plots.mm, left_margin=10Plots.mm,
               top_margin=6Plots.mm, right_margin=5Plots.mm,
               background_color=:white, foreground_color=:black,
               framestyle=:box)

for (idx, tz_shift) in enumerate(TIMEZONE_SHIFTS)
    idx > length(sleep_offsets_all) && break
    traces = sleep_offsets_all[idx]

    if isempty(traces)
        plot!(p_offset[idx], [], [], title = "$(tz_shift) h (No Data)",
              xlabel = "Days from shift", ylabel = "Sleep offset (h)",
              titlefont=("Times", 18, :bold), guidefont=("Times", 16), tickfont=("Times", 14),
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black, framestyle=:box)
        continue
    end

    available_indices = collect(1:length(traces))
    if PLOT_ONLY_STABILIZED
        available_indices = [i for i in available_indices if i <= length(stabilized_particles) && stabilized_particles[i]]
    end
    plot_indices = available_indices
    traces_to_plot = [traces[i] for i in plot_indices]

    for (j, trace_any) in enumerate(traces_to_plot)
        trace = collect(trace_any)
        n = length(trace)
        original_particle_idx = plot_indices[j]
        baseline_count = (original_particle_idx >= 1 && original_particle_idx <= length(baseline_cbtmin_counts)) ?
            baseline_cbtmin_counts[original_particle_idx] : BASELINE_DAYS
        day_zero_idx = baseline_count
        start_idx = max(1, day_zero_idx - 7)
        end_idx   = min(n, day_zero_idx + PLOT_POST_SHIFT_DAYS)
        (end_idx <= start_idx || day_zero_idx > n) && continue

        window = trace[start_idx:end_idx]
        x_unwrapped = unwrap24_centered(window)
        t = collect((start_idx:end_idx) .- day_zero_idx)

        plot!(p_offset[idx], t, x_unwrapped, alpha=trajectory_alpha, lw=0.8, color=trajectory_color,
              title="$(tz_shift) h", ylim=(-24, 24), xlim=(-7, PLOT_POST_SHIFT_DAYS),
              titlefont=("Times", 18, :bold), guidefont=("Times", 16), tickfont=("Times", 14),
              grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black, framestyle=:box)
    end

    plot!(p_offset[idx], [0, 0], [-24, 24], alpha=0.7, lw=2, color=:black, linestyle=:dash, xlim=(-7, PLOT_POST_SHIFT_DAYS), linewidth=2)
    plot!(p_offset[idx], [-7, PLOT_POST_SHIFT_DAYS], [0, 0], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_offset[idx], [-7, PLOT_POST_SHIFT_DAYS], [24, 24], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_offset[idx], [-7, PLOT_POST_SHIFT_DAYS], [-24, -24], alpha=0.4, lw=1, linestyle=:dash, color=:gray50)
    plot!(p_offset[idx], xlabel="Days relative to shift", ylabel="Sleep offset (h)",
          guidefont=("Times", 16), tickfont=("Times", 14), xguidefont=("Times", 16), yguidefont=("Times", 16))
end

plot!(p_offset, plot_title="Sleep Offset Trajectories Around Timezone Shifts\n(Window: day -7 to +$(PLOT_POST_SHIFT_DAYS) relative to shift)",
      plot_titlefontsize=24, titlefont=("Times", 24, :bold))
out_offset = joinpath(OUT_DIR, "sleep_offset_jetlag_trajectories$(filename_suffix).png")
savefig(p_offset, out_offset)
println("   Saved: $(out_offset)")

println("✅ All trajectory plots saved to $(OUT_DIR)")

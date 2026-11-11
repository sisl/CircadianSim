"""
CBTmin trajectories for timezone shifts 4, 9, 14, 19 h.
Two rows: 150 lux (top) and 10,000 lux (bottom); four equal columns; same width as `plotCombinedLightConditions.jl` (1400 px).
No figure title or per-panel titles (axes only). Shared **y** limits from all plotted (unwrapped) CBTmin values, with padding.

Lara Weed
"""

using CircadianPOMDP
using JLD2
using Plots
using Statistics
using StatsBase

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

const CONDITION_LOW = REGULAR_SCHEDULE_CONDITION_LOWLIGHT
const CONDITION_HIGH = REGULAR_SCHEDULE_CONDITION_HIGHLIGHT

const SELECTED_SHIFTS = [4, 9, 14, 19]
const N_PANELS = length(SELECTED_SHIFTS)

const FIG_WIDTH_PX = 1400
const FIG_HEIGHT_PX = round(Int, FIG_WIDTH_PX * 475 / 887)   # aspect ratio of the manuscript figure
# Manuscript style (figure_style.jl): bold axes, shift and light-condition labels inside the panels
const TICK_FS, GUIDE_FS, PANEL_FS = 17, 19, 21

const BASELINE_DAYS = BASELINE_PST_DAYS
# the analysis histories (150 lux: 100-day window, plotted to day 50)
const PLOT_POST_SHIFT_DAYS = TARGET_DAYS_FULL
const PLOT_ONLY_STABILIZED = true

# `CircadianPOMDP/Data/JetLag/3_JetLagAnalysis/` — same folder as `combined_light_conditions_summary.png`.
const OUT_DIR = joinpath(JETLAG_DATA_ROOT, DIR_JETLAG_ANALYSIS)

trajectory_color = :black
trajectory_alpha = 0.05
const ZERO_LINE_STYLE = (color=:black, alpha=0.85, lw=1.6, linestyle=:solid)

function reached_steady_state_by_end(cbtmin_trajectory, baseline_days=BASELINE_DAYS, min_consecutive_days=7, threshold=0.25)
    n = length(cbtmin_trajectory)
    n < baseline_days && return false
    baseline = cbtmin_trajectory[1:min(n, baseline_days)]
    return find_baseline_steady_state_cbtmin(baseline; min_consecutive_days, threshold) !== nothing
end

function load_cbtmin_trajectories_for_shift(tz_shift::Int, condition::String)
    filename = particle_hist_filename_for_shift(tz_shift; condition=condition, total_days=jetlag_total_days(condition))
    if !isfile(filename)
        println("   ⚠️  [$condition] $(tz_shift) h: missing — $(basename(filename))")
        return Vector{Vector{Float64}}()
    end
    hist = load_particle_histories(filename)
    cbtmins = events_to_trajectories(hist.cbtmin_events, hist.num_particles)
    println("   ✓ [$condition] $(tz_shift) h: $(hist.num_particles) particles")
    return cbtmins
end

function load_condition_bundle(condition::String)
    cbtmins_by_shift = Dict{Int, Vector{Vector{Float64}}}()
    for tz_shift in SELECTED_SHIFTS
        cbtmins_by_shift[tz_shift] = load_cbtmin_trajectories_for_shift(tz_shift, condition)
    end

    regular_histories_path = regular_histories_file(condition=condition)
    isfile(regular_histories_path) || error("Missing regular histories: $regular_histories_path")

    hist = load_particle_histories(regular_histories_path)
    baseline_hours = BASELINE_DAYS * 24
    baseline_cbtmin_trajectories, baseline_cbtmin_counts = baseline_cbtmin_from_events(
        hist.cbtmin_events, hist.num_particles, baseline_hours)

    stabilized_particles = BitVector()
    for (i, cbtmin_traj) in enumerate(baseline_cbtmin_trajectories)
        if isempty(cbtmin_traj)
            push!(stabilized_particles, false)
        else
            baseline_count = baseline_cbtmin_counts[i]
            if baseline_count > 0 && baseline_count <= length(cbtmin_traj)
                baseline_traj = cbtmin_traj[1:baseline_count]
                push!(stabilized_particles, reached_steady_state_by_end(baseline_traj, baseline_count, 7, 0.25))
            else
                push!(stabilized_particles, false)
            end
        end
    end
    included_set = Set(included_particle_ids())   # shared particle set (revision)
    stabilized_particles .&= [i in included_set for i in eachindex(stabilized_particles)]
    n_stab = count(stabilized_particles)
    println("   ✓ [$condition] stabilized by end of baseline: $(n_stab) / $(length(stabilized_particles)) particles")

    return (
        cbtmins = cbtmins_by_shift,
        baseline_cbtmin_counts = baseline_cbtmin_counts,
        stabilized_particles = stabilized_particles,
    )
end

println("📊 CBTmin jet lag trajectories — shifts $(SELECTED_SHIFTS) h")
println("   Rows: 150 lux (top) · 10,000 lux (bottom)")
println("   Figure: $(FIG_WIDTH_PX)×$(FIG_HEIGHT_PX) px, layout 2×$(N_PANELS)")
println()

println("Loading $(CONDITION_LOW)…")
bundle_low = load_condition_bundle(CONDITION_LOW)
println()
println("Loading $(CONDITION_HIGH)…")
bundle_high = load_condition_bundle(CONDITION_HIGH)
println()

mkpath(OUT_DIR)
Plots.default(fontfamily="Times", dpi=300, linewidth=2.5)

"""Collect every finite unwrapped CBTmin value that would be drawn (same filters as plotting)."""
function all_unwrapped_cbtmin_y(
    traces_low::Dict{Int, Vector{Vector{Float64}}},
    traces_high::Dict{Int, Vector{Vector{Float64}}},
    baseline_low,
    stab_low,
    baseline_high,
    stab_high,
)::Vector{Float64}
    ys = Float64[]
    rows = [(traces_low, baseline_low, stab_low), (traces_high, baseline_high, stab_high)]
    for (traces_by_shift, baseline_cbtmin_counts, stabilized_particles) in rows
        for tz_shift in SELECTED_SHIFTS
            traces = get(traces_by_shift, tz_shift, Vector{Vector{Float64}}())
            isempty(traces) && continue
            available_indices = collect(1:length(traces))
            if PLOT_ONLY_STABILIZED
                available_indices = [i for i in available_indices if i <= length(stabilized_particles) && stabilized_particles[i]]
            end
            plot_indices = available_indices
            for j in eachindex(plot_indices)
                trace = collect(traces[plot_indices[j]])
                n = length(trace)
                original_particle_idx = plot_indices[j]
                baseline_count = (original_particle_idx >= 1 && original_particle_idx <= length(baseline_cbtmin_counts)) ?
                    baseline_cbtmin_counts[original_particle_idx] : BASELINE_DAYS
                day_zero_idx = baseline_count
                start_idx = max(1, day_zero_idx - 7)
                end_idx = min(n, day_zero_idx + PLOT_POST_SHIFT_DAYS)
                (end_idx <= start_idx || day_zero_idx > n) && continue
                window = trace[start_idx:end_idx]
                x_unwrapped = unwrap24_centered(window)
                for v in x_unwrapped
                    isfinite(v) && push!(ys, v)
                end
            end
        end
    end
    return ys
end

"""Symmetric padding around data range; fallback if no points."""
function smart_ylim_from_values(ys::Vector{Float64})::Tuple{Float64,Float64}
    isempty(ys) && return (-24.0, 24.0)
    lo, hi = extrema(ys)
    !isfinite(lo) && return (-24.0, 24.0)
    span = hi - lo
    if span <= 1e-9
        δ = 2.0
        return (lo - δ, hi + δ)
    end
    pad = max(0.06 * span, 0.75)
    return (lo - pad, hi + pad)
end

"""In-panel labels as in the manuscript figure: shift (top right), light condition (first column), the two
re-entrainment routes (14 h, 10,000 lux) and the travel direction (4 h, 10,000 lux)."""
function add_panel_labels!(sp, row, idx, tz_shift, ylo, yhi)
    yat(f) = ylo + f * (yhi - ylo)
    annotate!(sp, PLOT_POST_SHIFT_DAYS - 1.5, yat(0.91), text("$(tz_shift)h", FIG_BOLD, PANEL_FS, :right))
    if idx == 1
        name, lux = row == 1 ? ("Low Light", "(150 lux)") : ("High Light", "(10,000 lux)")
        # centred at day 16 so the widest label, "(10,000 lux)", clears the dashed day-0 line
        annotate!(sp, 16, yat(0.91), text(name, FIG_BOLD, PANEL_FS - 2))
        annotate!(sp, 16, yat(0.81), text(lux, FIG_BOLD, PANEL_FS - 2))
    end
    if row == 2 && tz_shift == 14
        annotate!(sp, 22, 23, text("Phase Delay", FIG_BOLD, PANEL_FS - 3))
        annotate!(sp, 30, -10, text("Phase Advance", FIG_BOLD, PANEL_FS - 3))
    end
    if row == 2 && idx == 1
        annotate!(sp, 15, yat(0.30), text("New", FIG_BOLD, PANEL_FS - 3))
        annotate!(sp, 15, yat(0.21), text("Time Zone", FIG_BOLD, PANEL_FS - 3))
        plot!(sp, [2, 27], fill(yat(0.10), 2); color="#A6A6A6", linewidth=6, arrow=Plots.arrow(:closed, :head), label="")
    end
    return sp
end

function plot_trajectory_panels_two_rows!(
    p,
    traces_low::Dict{Int, Vector{Vector{Float64}}},
    traces_high::Dict{Int, Vector{Vector{Float64}}},
    baseline_low,
    stab_low,
    baseline_high,
    stab_high,
    ylabel_str::String,
    ylim_pair::Tuple{Float64,Float64},
)
    ylo, yhi = ylim_pair
    rows = [
        (traces_low, baseline_low, stab_low),
        (traces_high, baseline_high, stab_high),
    ]
    for (row, (traces_by_shift, baseline_cbtmin_counts, stabilized_particles)) in enumerate(rows)
        for (idx, tz_shift) in enumerate(SELECTED_SHIFTS)
            sp = (row - 1) * N_PANELS + idx
            traces = get(traces_by_shift, tz_shift, Vector{Vector{Float64}}())

            if isempty(traces)
                plot!(p[sp], [], [];
                    title="",
                    xlabel=(row == 2 ? "Days relative to shift" : ""),
                    ylabel=(idx == 1 ? ylabel_str : ""),
                    ylims=(ylo, yhi),
                    xlims=(-7, PLOT_POST_SHIFT_DAYS),
                    guidefont=("Times", 12), guidefontsize=12,
                    tickfont=("Times", 11), tickfontsize=11,
                    grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
                    framestyle=:box)
                continue
            end

            available_indices = collect(1:length(traces))
            if PLOT_ONLY_STABILIZED
                available_indices = [i for i in available_indices if i <= length(stabilized_particles) && stabilized_particles[i]]
            end
            plot_indices = available_indices
            traces_to_plot = [traces[i] for i in plot_indices]

            # Zero line (reviewer 2.7): solid, darker than the grid, under the trajectories
            if ylo <= 0 <= yhi
                plot!(p[sp], [-7, PLOT_POST_SHIFT_DAYS], [0, 0]; ZERO_LINE_STYLE...,
                    xlims=(-7, PLOT_POST_SHIFT_DAYS), ylims=(ylo, yhi))
            end

            for (j, trace_any) in enumerate(traces_to_plot)
                trace = collect(trace_any)
                n = length(trace)
                original_particle_idx = plot_indices[j]
                baseline_count = (original_particle_idx >= 1 && original_particle_idx <= length(baseline_cbtmin_counts)) ?
                    baseline_cbtmin_counts[original_particle_idx] : BASELINE_DAYS
                day_zero_idx = baseline_count
                start_idx = max(1, day_zero_idx - 7)
                end_idx = min(n, day_zero_idx + PLOT_POST_SHIFT_DAYS)
                (end_idx <= start_idx || day_zero_idx > n) && continue

                window = trace[start_idx:end_idx]
                x_unwrapped = unwrap24_centered(window)
                t = collect((start_idx:end_idx) .- day_zero_idx)

                plot!(p[sp], t, x_unwrapped, alpha=trajectory_alpha, lw=0.8, color=trajectory_color,
                    title="",
                    ylims=(ylo, yhi), xlims=(-7, PLOT_POST_SHIFT_DAYS),
                    guidefont=("Times", 12), guidefontsize=12,
                    tickfont=("Times", 11), tickfontsize=11,
                    grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
                    framestyle=:box)
            end

            plot!(p[sp], [0, 0], [ylo, yhi], alpha=0.7, lw=2, color=:black,
                linestyle=:dash, xlims=(-7, PLOT_POST_SHIFT_DAYS), ylims=(ylo, yhi), linewidth=2)

            plot!(p[sp],
                title="",
                xlabel=(row == 2 ? "Days After Travel" : ""),
                ylabel=(idx == 1 ? ylabel_str : ""),
                ylims=(ylo, yhi),
                xlims=(-7, PLOT_POST_SHIFT_DAYS),
                xformatter=(row == 2 ? :auto : (_ -> "")),
                yformatter=(idx == 1 ? :auto : (_ -> "")),
                top_margin=0Plots.mm, bottom_margin=(row == 2 ? 9Plots.mm : -2Plots.mm),
                left_margin=(idx == 1 ? 10Plots.mm : -2Plots.mm), right_margin=(idx == N_PANELS ? 4Plots.mm : 0Plots.mm),
                guidefont=(FIG_BOLD, GUIDE_FS), tickfont=(FIG_BOLD, TICK_FS),
                xguidefont=(FIG_BOLD, GUIDE_FS), yguidefont=(FIG_BOLD, GUIDE_FS))
            add_panel_labels!(p[sp], row, idx, tz_shift, ylo, yhi)
        end
    end
    return p
end

function make_cbtmin_figure(
    traces_low, traces_high,
    baseline_low, stab_low, baseline_high, stab_high,
    ylabel_str::String,
    stem::String,
)
    ys = all_unwrapped_cbtmin_y(traces_low, traces_high, baseline_low, stab_low, baseline_high, stab_high)
    ylim_pair = smart_ylim_from_values(ys)
    println("   Y limits (from data): $(round(ylim_pair[1], digits=2)) … $(round(ylim_pair[2], digits=2)) h ($(length(ys)) points)")

    p = plot(
        layout=grid(2, N_PANELS; widths=fill(1 / N_PANELS, N_PANELS), heights=fill(0.5, 2)),
        size=(FIG_WIDTH_PX, FIG_HEIGHT_PX),
        legend=false,
        plot_title="",
        margin=1Plots.mm,
        bottom_margin=4Plots.mm,
        left_margin=4Plots.mm,
        top_margin=2Plots.mm,
        right_margin=2Plots.mm,
        background_color=:white,
        foreground_color=:black,
        framestyle=:box,
    )
    plot_trajectory_panels_two_rows!(
        p,
        traces_low,
        traces_high,
        baseline_low,
        stab_low,
        baseline_high,
        stab_high,
        ylabel_str,
        ylim_pair,
    )

    suffix = PLOT_ONLY_STABILIZED ? "_stabilized_alpha005" : "_alpha005"
    fname = "$(stem)_4_9_14_19h_150lux_10klux$(suffix).png"
    out_file = joinpath(OUT_DIR, fname)
    savefig(p, out_file)
    println("✅ Saved: $(abspath(out_file))")
    return out_file
end

println("📈 Plotting 2×$(N_PANELS) CBTmin panels (day -7 … +$(PLOT_POST_SHIFT_DAYS))…")

make_cbtmin_figure(
    bundle_low.cbtmins,
    bundle_high.cbtmins,
    bundle_low.baseline_cbtmin_counts,
    bundle_low.stabilized_particles,
    bundle_high.baseline_cbtmin_counts,
    bundle_high.stabilized_particles,
    display_label("CBTmin (Hour)"),
    "cbtmin_jetlag_trajectories",
)

println()
println("Done. All figures in: $(abspath(OUT_DIR))")

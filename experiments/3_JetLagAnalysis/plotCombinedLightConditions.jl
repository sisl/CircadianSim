# Combined jet lag summary: LowLight150lux and HighLight10000lux on shared axes.
# Row 1: adaptation + twin axis; outer y guides only (left / right columns).
# Row 2: equal-width mean |SHAP| panels (no per-panel legend).
# Row 3: full-width legend (phase delay / advance / % advancing + SHAP features); no boxed outline.
#
# Prerequisites: detectSteadyState.jl for both conditions; summarizeAdaptationStatistics.jl
# run once per condition (to populate jetlag_shap_cache_path()), or SHAP panels will be empty.
#
# Lara Weed

using CircadianPOMDP  # before jetlag_reference.jl (defines JETLAG_MEAN_ALIGNMENT_SHIFT_H)
using JLD2
using Statistics
using Plots

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

const COND_LOW = REGULAR_SCHEDULE_CONDITION_LOWLIGHT
const COND_HIGH = REGULAR_SCHEDULE_CONDITION_HIGHLIGHT
const MIN_ADAPTATION_PLOT_FRACTION = 0.05

# SHAP variant from ARGS: "all" (default; parameters + regular-schedule outputs), "parameters" (static only)
# or "outputs" (regular-schedule timing only). Each feature keeps the same colour/marker in every variant,
# and all variants share one SHAP y-axis so the supplement panels are directly comparable.
const SHAP_VARIANTS = ("all", "parameters", "outputs")
const SHAP_VARIANT = (length(ARGS) >= 1 && ARGS[1] in SHAP_VARIANTS) ? ARGS[1] : "all"
const SHAP_VARIANT_RANGE = Dict("all" => 1:10, "parameters" => 1:5, "outputs" => 6:10)[SHAP_VARIANT]
# ψ fixed at 0 for every particle (standard Process C) is a constant column in the SHAP fits (zero importance) and is
# left out of the figure
const PSI_VARIES = length(unique(p.phi for p in load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"])) > 1
const SHAP_FEATURE_IDX = filter(i -> PSI_VARIES || i != 3, collect(SHAP_VARIANT_RANGE))
# Position of each plotted feature in the variant's importance vectors
const SHAP_VECTOR_POS = SHAP_FEATURE_IDX .- first(SHAP_VARIANT_RANGE) .+ 1
const SHAP_FEATURE_NAMES_ALL = ["τ", "k", "ψ", "χ", "μ_H/χ",
    "CBTmin", "Sleep Onset", "Sleep Offset", "Phase Offset", "Sleep Duration"]
# Legend text: Greek and subscripts as in the manuscript (figure_style.jl)
const SHAP_FEATURE_NAMES = display_label.(SHAP_FEATURE_NAMES_ALL[SHAP_FEATURE_IDX])
# Qualitative line colors: spaced hues, moderate chroma (easier to separate than grayed neighbors).
const SHAP_FEATURE_COLORS_ALL = [
    "#1565A8", "#00796B", "#6A1B9A", "#AD1457", "#C44F17",
    "#33691E", "#5D4037", "#00838F", "#6A4F2A", "#C62828",
]
const SHAP_FEATURE_LINESTYLES_ALL = [:solid, :dash, :dot, :dashdot, :solid,
    :dash, :dot, :dashdot, :solid, :dash]
const SHAP_FEATURE_MARKERS_ALL = [:circle, :square, :diamond, :utriangle, :dtriangle,
    :circle, :square, :diamond, :utriangle, :dtriangle]
const SHAP_FEATURE_COLORS = SHAP_FEATURE_COLORS_ALL[SHAP_FEATURE_IDX]
const SHAP_FEATURE_LINESTYLES = SHAP_FEATURE_LINESTYLES_ALL[SHAP_FEATURE_IDX]
const SHAP_FEATURE_MARKERS = SHAP_FEATURE_MARKERS_ALL[SHAP_FEATURE_IDX]

# Optional second argument "normalized": divide each shift's mean |SHAP| by the population mean adaptation time at
# that shift (SHAP decomposes predictions around their mean), so relative influence is comparable across shifts.
# Options after the variant (any order): "normalized" (above) and "noR2" (main-text layout: no R² strip; C–D
# carry the x axis). Supplement figures keep the R² strip.
const SHAP_OPTIONS = Set(ARGS[min(2, length(ARGS) + 1):end])
const SHAP_NORMALIZED = "normalized" in SHAP_OPTIONS
const SHOW_R2 = !("noR2" in SHAP_OPTIONS)
# "shaponly": supplement layout without the adaptation row; SHAP panels are lettered A–B and the legend lists
# the SHAP features only.
const SHAP_ONLY = "shaponly" in SHAP_OPTIONS
# "rawdays": plot the raw-day SHAP fits (supplementary check) instead of log(1 + days)
const SHAP_TARGET_PLOT = "rawdays" in SHAP_OPTIONS ? "days" : "logdays"
SHAP_NORMALIZED && SHAP_TARGET_PLOT == "logdays" &&
    error("'normalized' divides by mean days; log-day SHAP is already relative. Add 'rawdays' to normalize raw-day SHAP.")
const OUT_PATH = joinpath(JETLAG_DATA_ROOT, DIR_JETLAG_ANALYSIS,
    "combined_light_conditions_summary" * (SHAP_VARIANT == "all" ? "" : "_$(SHAP_VARIANT)") *
    (SHAP_NORMALIZED ? "_normalized" : "") * (SHOW_R2 ? "" : "_noR2") * (SHAP_ONLY ? "_shaponly" : "") * (SHAP_TARGET_PLOT == "days" ? "_rawdays" : "") * ".png")
const SHAP_PANEL_W = 690
const SHAP_PANEL_H = 400
const SHAP_Y_MAX = 1.05
const PHASE_DELAY_COLOR = "#0072B2"
const PHASE_ADVANCE_COLOR = "#E69F00"
const ADAPT_MEDIAN_LINEWIDTH = 4.0
const ADAPT_LINE_ALPHA = 1.0
const ADAPT_RIBBON_ALPHA = 0.24
const FIG_GRID_WIDTH = 1
const FIG_GRID_COLOR = :lightgray
const FIG_GRID_ALPHA = 0.6
# Left (days) axis maximum and grid step are set from the data once loaded (ADAPT_LEFT_Y_MAX below), so the
# 100-day window at 150 lux is not clipped; five steps keep the gridlines aligned with the right (%) axis
const ADAPT_RIGHT_Y_MAX = 100.0

# Times: axis/ticks use FIG_FONTSIZE (bold, as in the manuscript figure); legends slightly smaller.
const FIG_FONT = "Times"
const FIG_FONTSIZE = 18
# Light condition inside each panel (reviewer 2.8: 150 lux on A/C, 10,000 lux on B/D), top centre
const LUX_LABEL_FONTSIZE = 20
const LEGEND_FONTSIZE = 11
const PANEL_LETTER_FONTSIZE = 28
# Upper-left panel letters: below 0.95 so they clear the top spine (data coordinates).
const PANEL_LETTER_Y_FRAC = 0.88
const PANEL_TOP_MARGIN_MM = 6Plots.mm

"""Per-timezone stats for plotting; vectors align with `timezone_shifts` in results file."""
function adaptation_plot_vectors(results_data)
    steady_state_days_all = results_data["steady_state_days_by_timezone"]
    adaptation_directions_all = results_data["adaptation_directions_by_timezone"]
    timezone_shifts = results_data["timezone_shifts"]

    pct_delay = Float64[]
    pct_advance = Float64[]
    delay_medians = Float64[]
    delay_q25s = Float64[]
    delay_q75s = Float64[]
    advance_medians = Float64[]
    advance_q25s = Float64[]
    advance_q75s = Float64[]

    for (tz_idx, _) in enumerate(timezone_shifts)
        if tz_idx > length(steady_state_days_all) || isempty(steady_state_days_all[tz_idx])
            push!(pct_delay, NaN)
            push!(pct_advance, NaN)
            push!(delay_medians, NaN); push!(delay_q25s, NaN); push!(delay_q75s, NaN)
            push!(advance_medians, NaN); push!(advance_q25s, NaN); push!(advance_q75s, NaN)
            continue
        end

        adaptation_times = steady_state_days_all[tz_idx]
        directions = adaptation_directions_all[tz_idx]
        if length(adaptation_times) != length(directions)
            push!(pct_delay, NaN)
            push!(pct_advance, NaN)
            push!(delay_medians, NaN); push!(delay_q25s, NaN); push!(delay_q75s, NaN)
            push!(advance_medians, NaN); push!(advance_q25s, NaN); push!(advance_q75s, NaN)
            continue
        end

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
        n_forward = length(forward_times)
        n_backward = length(backward_times)
        push!(pct_delay, 100 * n_forward / n_total)
        push!(pct_advance, 100 * n_backward / n_total)

        if !isempty(forward_times) && length(forward_times) / n_total >= MIN_ADAPTATION_PLOT_FRACTION
            push!(delay_medians, median(forward_times))
            push!(delay_q25s, quantile(forward_times, 0.25))
            push!(delay_q75s, quantile(forward_times, 0.75))
        else
            push!(delay_medians, NaN)
            push!(delay_q25s, NaN)
            push!(delay_q75s, NaN)
        end

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

    return (timezone_shifts=timezone_shifts, pct_delay=pct_delay, pct_advance=pct_advance,
        delay_medians=delay_medians, delay_q25s=delay_q25s, delay_q75s=delay_q75s,
        advance_medians=advance_medians, advance_q25s=advance_q25s, advance_q75s=advance_q75s)
end

function load_shap_importance(condition::String; variant::String=SHAP_VARIANT)
    path = jetlag_shap_cache_path(condition=condition, variant=variant, target=SHAP_TARGET_PLOT)
    !isfile(path) && return nothing
    try
        d = load(path)
        # log(1 + days) fits are presented in days (first-order conversion, computeAdaptationSHAPVariants.jl)
        return get(d, SHAP_TARGET_PLOT == "logdays" ? "importance_days_by_tz" : "importance_by_tz", nothing)
    catch
        return nothing
    end
end

# Out-of-fold R² of the SHAP model per shift (computeAdaptationSHAPVariants.jl), drawn on a right-hand axis in C–D.
const R2_CSV = adaptation_r2_csv_path(target=SHAP_TARGET_PLOT)
const R2_COLUMN = Dict("all" => "r2_all_features", "parameters" => "r2_parameters", "outputs" => "r2_outputs")[SHAP_VARIANT]
const R2_COLOR = :gray35
const R2_LABEL = "Out-of-fold R²"

"""Shift → out-of-fold R² for `condition` and the current SHAP variant (empty if the CSV is missing)."""
function load_r2(condition::String)
    isfile(R2_CSV) || return Dict{Int,Float64}()
    lines = readlines(R2_CSV)
    hdr = split(lines[1], ",")
    ic, it, ir = findfirst(==("condition"), hdr), findfirst(==("tz"), hdr), findfirst(==(R2_COLUMN), hdr)
    out = Dict{Int,Float64}()
    for l in lines[2:end]
        f = split(l, ",")
        f[ic] == condition && (out[parse(Int, f[it])] = parse(Float64, f[ir]))
    end
    return out
end

"""Overlay R² by shift on a right-hand 0–1 axis (like % phase advancing in A–B)."""
function add_r2_axis!(p, r2::Dict{Int,Float64}; right_ylabel::String="")
    isempty(r2) && return p
    tz = sort(collect(keys(r2)))
    plot!(twinx(p[1]), tz, [r2[t] for t in tz],
        linewidth=2.0, linestyle=:dash, color=R2_COLOR, linealpha=0.75,
        marker=:diamond, markersize=4, markercolor=R2_COLOR, markeralpha=0.85, markerstrokewidth=0,
        label="", xlabel="", ylabel=right_ylabel, ylim=(0, 1), yticks=0:0.25:1, xlim=(0, 24),
        guidefont=(FIG_BOLD, FIG_FONTSIZE), tickfont=(FIG_BOLD, FIG_FONTSIZE),
        grid=false, legend=false)
    p[1][:framestyle] = :box   # twinx forces :axes; restore the full box
    return p
end

function shap_trends_panel(importance_by_tz; panel_w::Int=680, panel_h::Int=400, ymax::Float64=SHAP_Y_MAX,
        show_legend::Bool=true, right_margin=14Plots.mm, left_margin=11Plots.mm,
        bottom_margin=12Plots.mm, top_margin=PANEL_TOP_MARGIN_MM,
        outer_xlabel::Bool=true, outer_ylabel::Bool=true, show_xtick_labels::Bool=true, column::Symbol=:left)
    p = plot(size=(panel_w, panel_h),
        background_color=:white,
        foreground_color=:black,
        framestyle=:box,
        bottom_margin=bottom_margin,
        top_margin=top_margin,
        left_margin=col_margins(column).left_margin, right_margin=col_margins(column).right_margin,
        yformatter=(column == :left ? :auto : blank_ticks))

    x_lbl = outer_xlabel ? "Time Zone Shift (Hours)" : ""
    y_lbl = outer_ylabel ? (SHAP_NORMALIZED ? "Normalized Mean |SHAP|" : "Mean |SHAP| (Days)") : ""
    if importance_by_tz === nothing || isempty(importance_by_tz)
        plot!(p, framestyle=:box,
            xlim=(0, 24),
            ylim=(0, ymax),
            xticks=(4:4:24, string.(4:4:24)),
            xlabel=x_lbl,
            ylabel=y_lbl,
            guidefont=(FIG_BOLD, FIG_FONTSIZE),
            tickfont=(FIG_BOLD, FIG_FONTSIZE),
            grid=true,
            minorgrid=false,
            gridwidth=FIG_GRID_WIDTH,
            gridcolor=FIG_GRID_COLOR,
            gridalpha=FIG_GRID_ALPHA,
            legend=false)
        return p
    end

    valid_tz_shap = sort(collect(keys(importance_by_tz)))
    for (feat_idx, feat_name) in enumerate(SHAP_FEATURE_NAMES)
        importance_values = [importance_by_tz[tz][SHAP_VECTOR_POS[feat_idx]] for tz in valid_tz_shap]
        if feat_idx == 1
            plot!(p, valid_tz_shap, importance_values,
                linewidth=2.2,
                marker=SHAP_FEATURE_MARKERS[feat_idx],
                markersize=5,
                markeralpha=1.0,
                markerstrokewidth=0,
                markerstrokealpha=0,
                color=SHAP_FEATURE_COLORS[feat_idx],
                linestyle=SHAP_FEATURE_LINESTYLES[feat_idx],
                label=(show_legend ? feat_name : ""),
                xlabel=x_lbl,
                ylabel=y_lbl,
                guidefont=(FIG_BOLD, FIG_FONTSIZE),
                tickfont=(FIG_BOLD, FIG_FONTSIZE),
                legend=(show_legend ? (0.02, 0.98) : false),
                legend_column=(show_legend ? 2 : 1),
                legendfont=(FIG_FONT, LEGEND_FONTSIZE),
                legendfontsize=LEGEND_FONTSIZE,
                foreground_color_legend=nothing,
                xlim=(0, 24),
                xticks=(4:4:24, show_xtick_labels ? string.(4:4:24) : ["" for _ in 4:4:24]),
                ylim=(0, ymax),
                grid=true,
                minorgrid=false,
                gridwidth=FIG_GRID_WIDTH,
                gridcolor=FIG_GRID_COLOR,
                gridalpha=FIG_GRID_ALPHA,
                linealpha=0.88)
        else
            plot!(p, valid_tz_shap, importance_values,
                linewidth=2.2,
                marker=SHAP_FEATURE_MARKERS[feat_idx],
                markersize=5,
                markeralpha=1.0,
                markerstrokewidth=0,
                markerstrokealpha=0,
                color=SHAP_FEATURE_COLORS[feat_idx],
                linestyle=SHAP_FEATURE_LINESTYLES[feat_idx],
                label=(show_legend ? feat_name : ""),
                linealpha=0.88)
        end
    end
    vline!(p, [12],
        linestyle=:dash,
        linewidth=1.5,
        color=:gray40,
        alpha=0.6,
        label="")
    p
end

"""Full-width row below the two SHAP panels: adaptation path metrics + SHAP features (no legend box outline)."""
function combined_bottom_legend_row(;
        legend_columns::Int=4,
        row_h::Int=100,
        legend_fontsize::Int=FIG_FONTSIZE,
        top_margin=8Plots.mm,
)
    p = plot(size=(1400, row_h),
        showaxis=false,
        grid=false,
        framestyle=:none,
        background_color_subplot=:transparent,   # GR draws the x label slightly below its reserved space; a white strip would clip it
        margin=0.5Plots.mm,
        bottom_margin=1Plots.mm,
        top_margin=top_margin,
        xlim=(0, 24),
        ylim=(0, 1),
        legend=:top,   # directly under the panels, centred (within ~1 mm printed of the panel centre)
        legend_column=legend_columns,
        legendfont=(FIG_FONT, legend_fontsize),
        legendfontsize=legend_fontsize,
        foreground_color_legend=nothing,
        background_color_legend=:transparent,
        legend_foreground_color=:transparent)
    if !SHAP_ONLY
    plot!(p, [-1.0], [0.0],
        linewidth=ADAPT_MEDIAN_LINEWIDTH,
        color=PHASE_DELAY_COLOR,
        linestyle=:solid,
        marker=:circle,
        markersize=6,
        markeralpha=ADAPT_LINE_ALPHA,
        markerstrokewidth=0,
        label="Phase delay")
    plot!(p, [-1.0], [0.0],
        linewidth=ADAPT_MEDIAN_LINEWIDTH,
        color=PHASE_ADVANCE_COLOR,
        linestyle=:solid,
        marker=:square,
        markersize=6,
        markeralpha=ADAPT_LINE_ALPHA,
        markerstrokewidth=0,
        markerstrokealpha=0,
        label="Phase advance")
    plot!(p, [-1.0], [0.0],
        linewidth=2.2, linestyle=:solid, color=:gray60, linealpha=0.65,
        marker=:circle, markersize=4, markercolor=:gray60, markeralpha=0.82, markerstrokewidth=0,
        label="% phase advancing")
    end
    for feat_idx in LEGEND_ORDER
        feat_name = SHAP_FEATURE_NAMES[feat_idx]
        plot!(p, [-1.0], [0.0],
            linewidth=2.2,
            linestyle=SHAP_FEATURE_LINESTYLES[feat_idx],
            color=SHAP_FEATURE_COLORS[feat_idx],
            marker=SHAP_FEATURE_MARKERS[feat_idx],
            markersize=5,
            markercolor=SHAP_FEATURE_COLORS[feat_idx],
            markeralpha=1.0,
            markerstrokewidth=0,
            markerstrokealpha=0,
            label=feat_name)
    end
    p
end

# Columns: y tick labels on the outer sides only (left axis of the left column, right % axis of the right column),
# and narrow inner margins so the two columns sit close together; outer margins equal so the panels line up.
const OUTER_LEFT_MM, OUTER_RIGHT_MM, INNER_MM = 11Plots.mm, 14Plots.mm, 0Plots.mm   # GR keeps ~8.5 mm between columns
col_margins(column) = column == :left ? (left_margin=OUTER_LEFT_MM, right_margin=INNER_MM) :
                                        (left_margin=INNER_MM, right_margin=OUTER_RIGHT_MM)
# Bottom row (x tick labels + "Time Zone Shift (Hours)"): GR reserves the label space itself and adds bottom_margin
# below it, so a small margin keeps the legend close under the x label (~0.18 in printed between them). The canvas is
# shortened by what this saves relative to the earlier 12 mm, so the panels keep their size. The SHAP-only layouts
# give part of the saved space to the panels, so they keep more margin for the same gap.
const XAXIS_ROW_BOTTOM = SHAP_ONLY ? 5 : 1   # mm
const XAXIS_ROW_BOTTOM_MM = XAXIS_ROW_BOTTOM * Plots.mm
const XAXIS_ROW_SAVED_H = round(Int, (12 - XAXIS_ROW_BOTTOM) * 100 / 25.4)   # canvas units (100 per inch)
blank_ticks(_) = ""

"""Quarter-height strip: one series by shift (e.g. % phase advancing, out-of-fold R²) on its own y-axis."""
function strip_panel(tz, y; ylims, yticks, ylabel::String="", color=:gray60, linestyle=:solid, marker=:circle,
                     show_xtick_labels::Bool=false, xlabel::String="", panel_w::Int=520, panel_h::Int=130,
                     show_midline::Bool=true, column::Symbol=:left)
    tick_labels = show_xtick_labels ? string.(4:4:24) : ["" for _ in 4:4:24]
    p = plot(size=(panel_w, panel_h), background_color=:white, foreground_color=:black, framestyle=:box,
        margin=0Plots.mm, left_margin=col_margins(column).left_margin, right_margin=col_margins(column).right_margin, top_margin=0Plots.mm, yformatter=(column == :left ? :auto : blank_ticks),
        bottom_margin=(show_xtick_labels ? XAXIS_ROW_BOTTOM_MM : 0Plots.mm),
        xlims=(0, 24), ylims=ylims, yticks=yticks, xticks=(4:4:24, tick_labels), xlabel=xlabel, ylabel=ylabel,
        guidefont=(FIG_BOLD, FIG_FONTSIZE), tickfont=(FIG_BOLD, FIG_FONTSIZE), grid=true, gridalpha=0.25,
        legend=false)
    show_midline && vline!(p, [12], linestyle=:dash, linewidth=1.3, color=:gray40, alpha=0.55, label="")
    plot!(p, tz, y, linewidth=2.2, linestyle=linestyle, color=color, marker=marker, markersize=4,
        markercolor=color, markerstrokewidth=0, label="")
    return p
end

"""Per light level: phase delay vs phase advance (medians + IQR, left y) and % phase advancing (right y, twinx)."""
function adaptation_direction_panel(
    tz_axis,
    d_med, d_q25, d_q75,
    a_med, a_q25, a_q75,
    pct_advance::Vector{Float64};
    left_ylabel::String="",
    right_ylabel::String="",
    show_xtick_labels::Bool,
    show_path_legend::Bool=false,
    show_pct::Bool=true,
    column::Symbol=:left,
)
    p = plot(size=(520, 400),
        background_color=:white,
        foreground_color=:black,
        framestyle=:box,
        bottom_margin=(show_xtick_labels ? 12Plots.mm : 1Plots.mm),
        top_margin=(show_path_legend ? 10Plots.mm : PANEL_TOP_MARGIN_MM),
        left_margin=col_margins(column).left_margin, right_margin=col_margins(column).right_margin,
        guidefont=(FIG_BOLD, FIG_FONTSIZE),
        tickfont=(FIG_BOLD, FIG_FONTSIZE),
        xlims=(0, 24),
        ylims=(0, ADAPT_LEFT_Y_MAX),
    )

    tick_labels = show_xtick_labels ? string.(4:4:24) : ["" for _ in 4:4:24]

    plot!(p, tz_axis, a_q75, fillrange=a_q25, fillalpha=ADAPT_RIBBON_ALPHA, color=PHASE_ADVANCE_COLOR, linealpha=0, label="")
    plot!(p, tz_axis, d_q75, fillrange=d_q25, fillalpha=ADAPT_RIBBON_ALPHA, color=PHASE_DELAY_COLOR, linealpha=0, label="")

    h_y = collect(0:ADAPT_LEFT_GRID_STEP:Int(ADAPT_LEFT_Y_MAX))
    hline!(
        p,
        h_y;
        subplot=1,
        linecolor=FIG_GRID_COLOR,
        linewidth=FIG_GRID_WIDTH,
        linealpha=FIG_GRID_ALPHA,
        linestyle=:solid,
        primary=false,
        label="",
    )

    delay_label = show_path_legend ? "Delay" : ""
    advance_label = show_path_legend ? "Advance" : ""

    plot!(p, tz_axis, d_med,
        linewidth=ADAPT_MEDIAN_LINEWIDTH,
        color=PHASE_DELAY_COLOR,
        linestyle=:solid,
        marker=:circle,
        markersize=6,
        markeralpha=ADAPT_LINE_ALPHA,
        markerstrokewidth=0,
        label=delay_label,
        xlabel=show_xtick_labels ? "Time Zone Shift (Hours)" : "",
        ylabel=left_ylabel,
        grid=true,
        minorgrid=false,
        xgrid=true,
        ygrid=false,
        gridwidth=FIG_GRID_WIDTH,
        gridcolor=FIG_GRID_COLOR,
        gridalpha=FIG_GRID_ALPHA,
        xlim=(0, 24), xticks=(4:4:24, tick_labels),
        ylim=(0, ADAPT_LEFT_Y_MAX),
        yticks=0:ADAPT_LEFT_GRID_STEP:Int(ADAPT_LEFT_Y_MAX),
        yformatter=(column == :left ? :auto : blank_ticks),   # days axis labelled on the left column only
        linealpha=ADAPT_LINE_ALPHA,
        legend=(show_path_legend ? :topleft : false),
        legend_column=1,
        legendfontsize=LEGEND_FONTSIZE,
        background_color_legend=:transparent,
        legend_foreground_color=:transparent,
        legendfont=(FIG_FONT, LEGEND_FONTSIZE),
        foreground_color_legend=nothing)

    plot!(p, tz_axis, a_med,
        linewidth=ADAPT_MEDIAN_LINEWIDTH,
        color=PHASE_ADVANCE_COLOR,
        linestyle=:solid,
        marker=:square,
        markersize=6,
        markeralpha=ADAPT_LINE_ALPHA,
        markerstrokewidth=1,
        markerstrokecolor=:gray40,
        label=advance_label,
        linealpha=ADAPT_LINE_ALPHA)

    vline!(p, [12], linestyle=:dash, linewidth=1.3, color=:gray40, alpha=0.55, label="")

    if show_pct
        plot!(twinx(p[1]), tz_axis, pct_advance,
            linewidth=2.2,
            linestyle=:solid,
            color=:gray60,
            linealpha=0.65,
            marker=:circle,
            markersize=4,
            markercolor=:gray60,
            markeralpha=0.82,
            markerstrokewidth=0,
            label="",
            xlabel="",
            ylabel=right_ylabel,
            ylim=(0, ADAPT_RIGHT_Y_MAX),
            yticks=0:20:Int(ADAPT_RIGHT_Y_MAX),   # 5 steps, aligned with the left-axis gridlines (0:10:50),
            yformatter=(column == :right ? :auto : blank_ticks),   # % axis labelled on the right column only
            showaxis=(column == :right),   # left column: hidden, a blank axis would still reserve its label width
            left_margin=col_margins(column).left_margin, right_margin=col_margins(column).right_margin,   # twin axis too
            guidefont=(FIG_BOLD, FIG_FONTSIZE),
            tickfont=(FIG_BOLD, FIG_FONTSIZE),
            grid=false,
            legend=false)

        # twinx forces :axes (open corners); restore a full box on the primary subplot.
        p[1][:framestyle] = :box
    end

    p
end

# ——— Load steady-state results ———
path_low = steady_state_results_path(condition=COND_LOW)
path_high = steady_state_results_path(condition=COND_HIGH)
for (label, pth) in ((COND_LOW, path_low), (COND_HIGH, path_high))
    isfile(pth) || error("Missing steady-state results for $label: $pth\nRun detectSteadyState.jl $label")
end

data_low = load(path_low)
data_high = load(path_high)
tz_low = data_low["timezone_shifts"]
tz_high = data_high["timezone_shifts"]
tz_low == tz_high || error("timezone_shifts differ between conditions; cannot combine.")

println("📊 Combined light conditions (150 lux vs 10k lux), $(length(tz_low)) timezone shifts")

Plots.default(
    fontfamily=FIG_FONT,
    dpi=300,
    linewidth=2.5,
    guidefontsize=FIG_FONTSIZE,
    tickfontsize=FIG_FONTSIZE,
    legendfontsize=LEGEND_FONTSIZE,
)

low = adaptation_plot_vectors(data_low)
high = adaptation_plot_vectors(data_high)
_adapt_peak = maximum(filter(isfinite, vcat(low.delay_q75s, low.advance_q75s, high.delay_q75s, high.advance_q75s)))
const ADAPT_LEFT_Y_MAX = Float64(first(m for m in (50, 75, 100, 150) if m >= _adapt_peak))
const ADAPT_LEFT_GRID_STEP = Int(ADAPT_LEFT_Y_MAX) ÷ 5
println("   days axis 0–$(Int(ADAPT_LEFT_Y_MAX)) (largest plotted upper quartile $(_adapt_peak) d)")
tz_axis = low.timezone_shifts

# ——— Row 1: outer y labels on left / right columns only ———
p_150 = adaptation_direction_panel(
    tz_axis,
    low.delay_medians, low.delay_q25s, low.delay_q75s,
    low.advance_medians, low.advance_q25s, low.advance_q75s,
    low.pct_advance;
    left_ylabel="Adaptation Time (Days)",
    right_ylabel="",
    show_xtick_labels=false,
    show_path_legend=false)

p_10k = adaptation_direction_panel(
    tz_axis,
    high.delay_medians, high.delay_q25s, high.delay_q75s,
    high.advance_medians, high.advance_q25s, high.advance_q75s,
    high.pct_advance;
    left_ylabel="",
    right_ylabel="% Phase Advancing",
    show_xtick_labels=false,
    column=:right)

# ——— SHAP: equal-width panels; shared bottom legend row ———
"""Population mean adaptation time (days) per shift; non-adapters at the window length (restricted mean)."""
mean_days_by_tz(data) = Dict(tz => mean(vcat(Float64.(d), fill(Float64(data["post_shift_window_days"]), length(n))))
                             for (tz, d, n) in zip(data["timezone_shifts"], data["steady_state_days_by_timezone"],
                                                   data["not_adapted_particle_ids_by_timezone"]) if !isempty(d))
const MEAN_DAYS = Dict(COND_LOW => mean_days_by_tz(data_low), COND_HIGH => mean_days_by_tz(data_high))
"""Importance by shift, optionally divided by the mean adaptation time at that shift."""
function shap_for_plot(c; variant=SHAP_VARIANT)
    imp = load_shap_importance(c; variant)
    (imp === nothing || !SHAP_NORMALIZED) && return imp
    return Dict(tz => v ./ MEAN_DAYS[c][tz] for (tz, v) in imp if haskey(MEAN_DAYS[c], tz))
end
imp_low = shap_for_plot(COND_LOW)
imp_high = shap_for_plot(COND_HIGH)

# Shared y-axis for C and D, scaled to the data (a fixed 1.05 h cap clipped curves once rows were joined by id, #27)
# The scale is shared across all variants (whichever caches exist), so the three supplement figures match.
shap_max(imp) = (imp === nothing || isempty(imp)) ? 0.0 : maximum(maximum(v) for v in values(imp))
_shap_peak = maximum(shap_max(shap_for_plot(c; variant=v)) for c in (COND_LOW, COND_HIGH), v in SHAP_VARIANTS)
# Normalised panels: round up with headroom so the panel letter clears the curves
SHAP_YMAX_DATA = SHAP_NORMALIZED ? ceil(1.3 * _shap_peak / 0.1) * 0.1 :
                 SHAP_TARGET_PLOT == "logdays" ? ceil(1.15 * _shap_peak) : max(SHAP_Y_MAX, 1.05 * _shap_peak)   # whole days
p_shap_low = shap_trends_panel(imp_low; panel_w=SHAP_PANEL_W, panel_h=SHAP_PANEL_H, ymax=SHAP_YMAX_DATA,
    show_legend=false, outer_xlabel=!SHOW_R2, outer_ylabel=true, top_margin=(SHAP_ONLY ? PANEL_TOP_MARGIN_MM : 1Plots.mm),
    bottom_margin=(SHOW_R2 ? 1Plots.mm : XAXIS_ROW_BOTTOM_MM), show_xtick_labels=!SHOW_R2, column=:left)
p_shap_high = shap_trends_panel(imp_high; panel_w=SHAP_PANEL_W, panel_h=SHAP_PANEL_H, ymax=SHAP_YMAX_DATA,
    show_legend=false, outer_xlabel=!SHOW_R2, outer_ylabel=false, top_margin=(SHAP_ONLY ? PANEL_TOP_MARGIN_MM : 1Plots.mm),
    bottom_margin=(SHOW_R2 ? 1Plots.mm : XAXIS_ROW_BOTTOM_MM), show_xtick_labels=!SHOW_R2, column=:right)
# Quarter-height R² strip below C–D (supplement figures)
r2_vec(c) = (r = load_r2(c); [get(r, t, NaN) for t in tz_axis])
r2_strip(c, ylabel, column) = strip_panel(tz_axis, r2_vec(c); ylims=(0, 1), yticks=0:0.5:1, ylabel=ylabel,
                                  color=R2_COLOR, linestyle=:dash, marker=:diamond,
                                  show_xtick_labels=true, xlabel="Time Zone Shift (Hours)", column=column)
p_r2_low, p_r2_high = r2_strip(COND_LOW, "R²", :left), r2_strip(COND_HIGH, "", :right)
# Legend row sized to its entries so none is cut off. GR gives every legend column the width of the longest entry, so
# the legend is centred only if a longest entry ends a row: 3 columns with the adaptation entries ("% phase advancing"
# last) on the first row, or the five parameters on one row in the parameters SHAP-only layout (longer output names
# would overflow one row, so outputs use 3 columns).
const N_LEGEND = (SHAP_ONLY ? 0 : 3) + length(SHAP_FEATURE_NAMES)
const LEGEND_COLS = (SHAP_ONLY && SHAP_VARIANT == "parameters") ? N_LEGEND : 3
# Outputs-only SHAP legend (3 columns): longest name ("Sleep Duration") at the end of the first row so it is centred
const LEGEND_ORDER = (SHAP_ONLY && SHAP_VARIANT == "outputs") ? [1, 2, 5, 3, 4] : collect(eachindex(SHAP_FEATURE_NAMES))
const LEGEND_H = 30 + 30 * cld(N_LEGEND, LEGEND_COLS)   # canvas units (1400 wide)
p_bottom_legend = combined_bottom_legend_row(; legend_columns=LEGEND_COLS, top_margin=2Plots.mm)

# Panel letters (compareWithNHANES.jl style: upper left in data coordinates).
# subplot=1 only: twinx adds a second subplot and annotate! would otherwise place the letter twice.
annotate!(p_150, 0.05 * 24, PANEL_LETTER_Y_FRAC * ADAPT_LEFT_Y_MAX, text("A", FIG_FONT, PANEL_LETTER_FONTSIZE); subplot=1)
annotate!(p_10k, 0.05 * 24, PANEL_LETTER_Y_FRAC * ADAPT_LEFT_Y_MAX, text("B", FIG_FONT, PANEL_LETTER_FONTSIZE); subplot=1)
annotate!(p_shap_low, 0.05 * 24, PANEL_LETTER_Y_FRAC * SHAP_YMAX_DATA, text(SHAP_ONLY ? "A" : "C", FIG_FONT, PANEL_LETTER_FONTSIZE))
annotate!(p_shap_high, 0.05 * 24, PANEL_LETTER_Y_FRAC * SHAP_YMAX_DATA, text(SHAP_ONLY ? "B" : "D", FIG_FONT, PANEL_LETTER_FONTSIZE))
for (pp, x, lux, ymax) in ((p_150, 11.5, "150 lux", ADAPT_LEFT_Y_MAX), (p_10k, 10.5, "10,000 lux", ADAPT_LEFT_Y_MAX),
                           (p_shap_low, 11.5, "150 lux", SHAP_YMAX_DATA), (p_shap_high, 10.5, "10,000 lux", SHAP_YMAX_DATA))
    annotate!(pp, x, PANEL_LETTER_Y_FRAC * ymax, text(lux, FIG_BOLD, LUX_LABEL_FONTSIZE); subplot=1)
end

# 2×2 grid with link=:both aligns x within columns and y within rows. A separate legend
# row avoids linking y across data panels and the legend strip (which would collapse scales).
# Rows: A–B (adaptation days) | % phase advancing (¼ height) | C–D (SHAP) | out-of-fold R² (¼ height, x axis)
# % phase advancing is on the right-hand axis of A–B; the R² strip (supplement only) carries the x axis.
shap_rows = SHOW_R2 ? (p_shap_low, p_shap_high, p_r2_low, p_r2_high) : (p_shap_low, p_shap_high)
panels = SHAP_ONLY ? shap_rows : (p_150, p_10k, shap_rows...)
data_layout, data_h = if SHAP_ONLY
    SHOW_R2 ? (grid(2, 2; heights=[0.74, 0.26]), 690) : ((1, 2), 560)   # with the legend ≤ 4.25 in tall at 7.5 in (half page)
else
    SHOW_R2 ? (grid(3, 2; heights=[0.44, 0.41, 0.15]), 1050) : ((2, 2), 925)   # main text: original Fig 4 size
end
p_data = plot(
    panels...;
    layout=data_layout,
    link=:both,
    size=(1400, data_h),
    dpi=300,   # margins come from each panel (a bottom_margin here would override all of them)
)
data_total_h = (SHAP_ONLY ? data_h : (SHOW_R2 ? 1023 : 887)) - XAXIS_ROW_SAVED_H
fig_h = data_total_h + LEGEND_H
combined = plot(
    p_data,
    p_bottom_legend,
    layout=grid(2, 1; heights=[data_total_h / fig_h, LEGEND_H / fig_h]),
    size=(1400, fig_h),
    dpi=300,
    link=:none,
)

mkpath(dirname(OUT_PATH))
savefig(combined, OUT_PATH)
println("✅ Saved: $OUT_PATH")

"""
Wrapped Plot: Baseline vs Best Intervention (Unified steady-state data)
Reads data saved by rescuingJetLag_unified.jl (systematic_rescue_analysis_steadyState.jld2).
Adaptation times are those computed in the unified script (steady-state method, same as
experiments/3_JetLagAnalysis/detectSteadyState.jl).
Shows Low Light (top) and High Light (bottom) when both are present; 24-hour periodicity.
Main figure: Baseline; Oracle personal best (each participant's best policy or no intervention) and Oracle personal
worst (each participant's worst light window). With ARGS "general" (supplementary file *_withGeneral.png) also Oracle
general best / worst: one window per timezone for everyone, chosen with all participants' simulated outcomes (the one
making the most participants faster / slower than without intervention; general_policy_key and general_worst_key in
jetlag_load_steady_state.jl).
Median as line, quartiles (25th–75th percentiles) as shaded region. Participants not adapted within the post-shift
window count as the window length (right-censored; see jetlag_load_steady_state.jl).
Also writes the per-shift numbers (incl. % not adapted) to baseline_vs_intervention_summary_steadyState.csv.
Colors: colour-blind-safe Okabe–Ito palette (black / grey / blues / oranges) plus distinct line styles for grayscale.
Values at ±24 and 0 set to zero.
Lara Weed
February 2025
"""

println("🌍 Creating Wrapped Baseline vs Best Intervention Plot (Unified steady-state, quartiles + oracle best/worst)")
println("📦 Setting up packages...")

using CircadianPOMDP
using JLD2
using Plots
using Statistics
using Printf
using Dates

# Times (serif) for all plot text. Width below matches ~12 cm at 300 dpi so printed pt sizes are trustworthy.
const WRAPPED_FIG_WIDTH_CM = 12.0
const WRAPPED_FIG_DPI = 300
const WRAPPED_FIG_WIDTH_PX = round(Int, WRAPPED_FIG_WIDTH_CM / 2.54 * WRAPPED_FIG_DPI)  # ≈ 1417
# Font sizes (pt): sized for single-column ~12 cm width (tick ≥11–12 pt is a common journal minimum).
default(fontfamily="Times",
        titlefont=("Times", 16, :bold),
        guidefont=("Times", 14, :bold),
        tickfont=("Times", 12),
        legendfont=("Times", 13, :bold),
        plot_titlefont=("Times", 18, :bold),
        dpi=WRAPPED_FIG_DPI)

# HypothesisTests for normality (Shapiro-Wilk)
try
    using HypothesisTests
    global HAS_HYPOTHESISTESTS = true
catch
    @warn "HypothesisTests not available. Normality tests will be skipped."
    global HAS_HYPOTHESISTESTS = false
end

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "jetlag_load_steady_state.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

println("📦 All packages loaded successfully!")
println()

"""Run Shapiro-Wilk normality test on data; return (p_value, statistic, normal?)."""
function test_normality_shapiro(data::Vector{Float64})
    if !HAS_HYPOTHESISTESTS || length(data) < 3
        return (nothing, nothing, nothing)
    end
    try
        t = HypothesisTests.ShapiroWilkTest(data)
        return (pvalue(t), t.W, pvalue(t) > 0.05)
    catch
        return (nothing, nothing, nothing)
    end
end

# Load conditions that exist (High Light and/or Low Light)
println("📊 Loading unified steady-state data (adaptation times from rescuingJetLag_unified.jl)...")

conditions_to_load = [
    ("LowLight150lux", "Low Light (150 lux)"),
    ("HighLight10000lux", "High Light (10,000 lux)"),
]
loaded = []
raw_for_normality = []  # (plot_title, timezones, general_best_raw, personalized_raw)
summary_rows = String[]  # per-shift table written next to the figure
worst_by_condition = Dict{String, Any}()    # oracle personal worst (median, q25, q75) per condition
gworst_by_condition = Dict{String, Any}()   # oracle general worst (median, q25, q75) per condition
for (condition_name, plot_title) in conditions_to_load
    try
        timezones, baseline_times, baseline_ci_low, baseline_ci_high,
        intervention_times, intervention_ci_low, intervention_ci_high,
        personalized_times, personalized_ci_low, personalized_ci_high,
        general_best_raw, personalized_raw, baseline_raw, details =
            load_unified_steady_state_data(condition_name; return_raw_times=true, return_details=true)
        if !isempty(timezones)
            push!(loaded, (condition_name, plot_title, timezones, baseline_times, baseline_ci_low, baseline_ci_high,
                intervention_times, intervention_ci_low, intervention_ci_high,
                personalized_times, personalized_ci_low, personalized_ci_high))
            push!(raw_for_normality, (plot_title, timezones, general_best_raw, personalized_raw))
            worst_by_condition[condition_name] = (details.worst_median, details.worst_q25, details.worst_q75)
            gworst_by_condition[condition_name] = (details.general_worst_median, details.general_worst_q25, details.general_worst_q75)
            println("   ✓ $(plot_title): $(length(timezones)) timezone shifts; % not adapted within " *
                    "$(details.window_days) d (baseline) $(round(minimum(details.pct_not_adapted_baseline); digits=1))–" *
                    "$(round(maximum(details.pct_not_adapted_baseline); digits=1))")
            for (i, tz) in enumerate(timezones)
                gp = details.general_policy[i]
                push!(summary_rows, join([condition_name, tz, details.window_days,
                    baseline_times[i], baseline_ci_low[i], baseline_ci_high[i], round(details.pct_not_adapted_baseline[i]; digits=2),
                    intervention_times[i], intervention_ci_low[i], intervention_ci_high[i], round(details.pct_not_adapted_general[i]; digits=2),
                    personalized_times[i], personalized_ci_low[i], personalized_ci_high[i], round(details.pct_not_adapted_personalized[i]; digits=2),
                    gp === nothing ? "" : gp.start, gp === nothing ? "" : gp.duration, details.n_tied[i],
                    round(details.pct_helped_general[i]; digits=2), round(details.pct_harmed_general[i]; digits=2),
                    details.worst_median[i], details.worst_q25[i], details.worst_q75[i], round(details.pct_slowed_worst[i]; digits=2),
                    details.general_worst_median[i], details.general_worst_q25[i], details.general_worst_q75[i],
                    round(details.pct_slowed_general_worst[i]; digits=2),
                    (gw = details.general_worst_policy[i]; gw === nothing ? "" : gw.start), (gw === nothing ? "" : gw.duration)], ","))
            end
        end
    catch e
        println("   ⏭ Skipping $(plot_title) (no data)")
        println("      Error: $(sprint(showerror, e))")
    end
end

if isempty(loaded)
    error("No condition data found. Run rescuingJetLag_unified.jl first.")
end
println("   Plotting $(length(loaded)) condition(s)")

# ================================================================
# NORMALITY TESTS: Best General and Oracle at each timezone
# ================================================================
println()
println("📊 Normality tests (Shapiro-Wilk, α=0.05): Best General vs Oracle per timezone")
if HAS_HYPOTHESISTESTS
    for (plot_title, tzs, general_best_raw, personalized_raw) in raw_for_normality
        println("   Condition: $(plot_title)")
        general_normal = 0
        general_nonnormal = 0
        personal_normal = 0
        personal_nonnormal = 0
        for tz in tzs
            g_raw = filter(isfinite, get(general_best_raw, tz, Float64[]))
            p_raw = filter(isfinite, get(personalized_raw, tz, Float64[]))
            g_p, g_W, g_ok = test_normality_shapiro(g_raw)
            p_p, p_W, p_ok = test_normality_shapiro(p_raw)
            g_str = g_ok === nothing ? "N/A" : (g_ok ? "normal (p=$(round(g_p; digits=4)))" : "non-normal (p=$(round(g_p; digits=4)))")
            p_str = p_ok === nothing ? "N/A" : (p_ok ? "normal (p=$(round(p_p; digits=4)))" : "non-normal (p=$(round(p_p; digits=4)))")
            println("      TZ $(lpad(tz, 3)) h: General=$(g_str)  |  Oracle=$(p_str)")
            g_ok === true && (general_normal += 1)
            g_ok === false && (general_nonnormal += 1)
            p_ok === true && (personal_normal += 1)
            p_ok === false && (personal_nonnormal += 1)
        end
        println("   Summary: General best — $(general_normal) normal, $(general_nonnormal) non-normal  |  Oracle — $(personal_normal) normal, $(personal_nonnormal) non-normal")
        println()
    end
else
    println("   ⏭ HypothesisTests not available. Install with: using Pkg; Pkg.add(\"HypothesisTests\")")
end

# ================================================================
# CREATE WRAPPED DATA WITH EXTENDED X-AXIS (-24 to +24)
# ================================================================

println("📊 Creating wrapped data with extended x-axis...")

extended_x = collect(-24:24)

function create_wrapped_data(original_x, original_y, original_ci_low, original_ci_high)
    extended_y = Float64[]
    extended_ci_low = Float64[]
    extended_ci_high = Float64[]

    for x in extended_x
        if x == 0 || abs(x) == 24
            push!(extended_y, 0.0)
            push!(extended_ci_low, 0.0)
            push!(extended_ci_high, 0.0)
        else
            wrapped_x = x > 0 ? x : (24 + x)
            wrapped_x = wrapped_x == 0 ? 24 : wrapped_x

            if wrapped_x in original_x
                idx = findfirst(==(wrapped_x), original_x)
                push!(extended_y, original_y[idx])
                push!(extended_ci_low, original_ci_low[idx])
                push!(extended_ci_high, original_ci_high[idx])
            else
                if isempty(original_x)
                    push!(extended_y, 0.0)
                    push!(extended_ci_low, 0.0)
                    push!(extended_ci_high, 0.0)
                else
                    if wrapped_x < minimum(original_x)
                        idx = 1
                    elseif wrapped_x > maximum(original_x)
                        idx = length(original_x)
                    else
                        distances = abs.(original_x .- wrapped_x)
                        idx = argmin(distances)
                    end
                    push!(extended_y, original_y[idx])
                    push!(extended_ci_low, original_ci_low[idx])
                    push!(extended_ci_high, original_ci_high[idx])
                end
            end
        end
    end

    return extended_y, extended_ci_low, extended_ci_high
end

# Extended (wrapped) data per loaded condition
extended_data = []
for (cname, _, tz, bt, bcl, bch, it, icl, ich, pt, pcl, pch) in loaded
    eb, eb_lo, eb_hi = create_wrapped_data(tz, bt, bcl, bch)
    ei, ei_lo, ei_hi = create_wrapped_data(tz, it, icl, ich)
    ep, ep_lo, ep_hi = create_wrapped_data(tz, pt, pcl, pch)
    ew, ew_lo, ew_hi = create_wrapped_data(tz, worst_by_condition[cname]...)
    eg, eg_lo, eg_hi = create_wrapped_data(tz, gworst_by_condition[cname]...)
    push!(extended_data, (baseline=(eb, eb_lo, eb_hi), intervention=(ei, ei_lo, ei_hi), personalized=(ep, ep_lo, ep_hi),
                          worst=(ew, ew_lo, ew_hi), general_worst=(eg, eg_lo, eg_hi)))
end

println("   ✓ Extended data from -24 to +24 hours for $(length(loaded)) condition(s)")

# ================================================================
# CREATE WRAPPED PLOT WITH SHADED REGIONS
# ================================================================

println("📊 Creating wrapped baseline vs intervention plot (steady-state, quartiles)...")

# Colour-blind safe (Okabe–Ito; no green): best = blues, worst = oranges, i.e. the blue–orange axis that red–green
# colour vision deficiency preserves; personal = dark/saturated + dotted, general = light + dashed, so every pair also
# differs in lightness and line style.
# Main figure: baseline, oracle personal best and oracle personal worst. Pass "general" to add the oracle general
# best / worst (one window per shift for everyone, chosen with all participants' simulated outcomes; separate file);
# best and worst then go in separate columns (baseline in both) so each panel has at most three interquartile bands
# and every band and median line stays visible ("combined" puts all five in one panel per condition).
const SHOW_GENERAL = "general" in ARGS
const SPLIT = SHOW_GENERAL && !("combined" in ARGS)
const COL_BASELINE = "#000000"        # black (line)
const COL_BASELINE_FILL = "#7F7F7F"   # neutral grey (band)
const COL_GENERAL  = "#56B4E9"        # sky blue (oracle general best)
const COL_PERSONAL = "#0072B2"        # blue (oracle personal best)
const COL_GENERAL_WORST = "#E69F00"   # orange (oracle general worst)
const COL_WORST    = "#D55E00"        # vermillion (oracle personal worst)
# Per-series band opacity (interquartile ranges, no outlines)
const FILL_BASELINE, FILL_GENERAL, FILL_PERSONAL = 0.30, 0.30, 0.25
band!(p, x, lo, hi, col, fill_alpha, sp) =
    plot!(p, x, lo, fillrange=hi, fillcolor=col, fillalpha=fill_alpha, linewidth=0, linecolor=col, label="", subplot=sp)
const MEDIAN_LINEWIDTH = 4.0
const LINE_ALPHA = 1.0
const LS_BASELINE = :solid
const LS_GENERAL  = :dash
const LS_PERSONAL = :dot

"""Curves (median, q25, q75) with colour, band opacity, line style and label: (best, worst) for one condition."""
curves(ext) = (best=[(ext.intervention, COL_GENERAL, FILL_GENERAL, LS_GENERAL, "Oracle general best", true),
                     (ext.personalized, COL_PERSONAL, FILL_PERSONAL, LS_PERSONAL, "Oracle personal best", false)],
               worst=[(ext.general_worst, COL_GENERAL_WORST, FILL_GENERAL, LS_GENERAL, "Oracle general worst", true),
                      (ext.worst, COL_WORST, FILL_PERSONAL, LS_PERSONAL, "Oracle personal worst", false)])

# In the narrower split panels GR draws shorter legend samples that touch the text; a leading space separates them
legend_label(s) = SPLIT ? "  " * s : s

"""Baseline band, the curves' bands, then the median lines (baseline first, personal on top), and the wrap markers."""
function draw_panel!(p, sp, ext, cs)
    cs = filter(c -> SHOW_GENERAL || !c[6], cs)
    eb, eb_lo, eb_hi = ext.baseline
    band!(p, extended_x, eb_lo, eb_hi, COL_BASELINE_FILL, FILL_BASELINE, sp)
    for ((_, lo, hi), col, fa, _, _, _) in cs
        band!(p, extended_x, lo, hi, col, fa, sp)
    end
    plot!(p, extended_x, eb, subplot=sp, label=legend_label("Baseline"), linewidth=MEDIAN_LINEWIDTH,
          color=COL_BASELINE, linestyle=LS_BASELINE, alpha=LINE_ALPHA)
    for ((m, _, _), col, _, ls, lab, _) in cs
        plot!(p, extended_x, m, subplot=sp, label=legend_label(lab), linewidth=MEDIAN_LINEWIDTH, color=col, linestyle=ls, alpha=LINE_ALPHA)
    end
    # Midpoint / wrap markers: neutral gray so they are not confused with baseline (black).
    vline!(p, [0], color="#555555", linestyle=:dash, alpha=0.65, linewidth=1.2, label="", subplot=sp)
    vline!(p, [-24, 24], color="#888888", linestyle=:dot, alpha=0.55, linewidth=1, label="", subplot=sp)
    return vcat(eb_hi, [c[1][3] for c in cs]...)
end

n_sub = length(loaded)
n_col = SPLIT ? 2 : 1
# Both layouts are drawn at the same width (scaled to the page width when printed); split: two columns,
# each row 0.36 × the width high.
fig_w_px = WRAPPED_FIG_WIDTH_PX   # same width as the main figure, so text prints at the same size
# Main figure: aspect ratio of the manuscript figure (881 × 581 px as assembled, two rows)
fig_h_px = SPLIT ? round(Int, WRAPPED_FIG_WIDTH_PX * 0.36 * n_sub) : round(Int, WRAPPED_FIG_WIDTH_PX * 581 / 881 * n_sub / 2)
p = plot(layout=(n_sub, n_col), size=(fig_w_px, fig_h_px), dpi=WRAPPED_FIG_DPI,
         background_color=:white, foreground_color=:black,
         fontfamily="Times",
         left_margin=(SPLIT ? 9 : 5) * Plots.mm, right_margin=(SPLIT ? 2 : 3) * Plots.mm,
         top_margin=(SPLIT ? 3 : 1) * Plots.mm, bottom_margin=(SPLIT ? 9 : 6) * Plots.mm)

all_ci_high = Float64[]
panel_titles = String[]   # in subplot order (row-major)
for ((_, plot_title, _, _, _, _, _, _, _, _, _, _), ext) in zip(loaded, extended_data)
    c = curves(ext)
    if SPLIT
        append!(all_ci_high, draw_panel!(p, length(panel_titles) + 1, ext, c.best)); push!(panel_titles, "$(plot_title): best timing")
        append!(all_ci_high, draw_panel!(p, length(panel_titles) + 1, ext, c.worst)); push!(panel_titles, "$(plot_title): worst timing")
    else
        append!(all_ci_high, draw_panel!(p, length(panel_titles) + 1, ext, vcat(c.best, c.worst))); push!(panel_titles, plot_title)
    end
end
filter!(isfinite, all_ci_high)

finite_ci = filter(isfinite, all_ci_high)
common_ylims = (0, (isempty(finite_ci) ? 50.0 : maximum(finite_ci)) * 1.1)
max_days = ceil(Int, common_ylims[2])
# X/Y: tick marks only at labeled positions so grid lines are not drawn every 1 h / 1 day.
xtick_positions = collect(-24:(SPLIT ? 8 : 4):24)
xtick_labels = string.(xtick_positions)
ytick_positions = collect(0:(max_days > 40 ? 10 : 5):max_days)
ytick_labels = string.(ytick_positions)

# Manuscript style (figure_style.jl): bold axes; main figure has the light condition inside each panel (two lines,
# top left), East/West and "Longer Jet Lag" arrows in the top panel, and x ticks on the bottom panel only.
const LABEL_FS, GUIDE_FS, TICK_FS, LEGEND_FS = SPLIT ? (21, 20, 18, 18) : (20, 19, 17, 16)   # main at 7.5 in: 10.6, 10.1, 9.0, 8.5 pt; SI split at 6.5 in: 9.6, 9.2, 8.3, 8.3 pt
for (sp, panel_title) in enumerate(panel_titles)
    bottom_row = sp > (n_sub - 1) * n_col
    plot!(p, subplot=sp,
          title=(SPLIT ? panel_title : ""),
          titlefont=(FIG_BOLD, LABEL_FS),
          ylabel=(mod1(sp, n_col) == 1 ? "Adaptation Time (Days)" : ""),
          yguidefont=(FIG_BOLD, GUIDE_FS),
          xlims=(-24, 24), ylims=common_ylims,
          xticks=(xtick_positions, bottom_row || SPLIT ? xtick_labels : fill("", length(xtick_positions))),
          yticks=(ytick_positions, ytick_labels),
          xtickfont=(FIG_BOLD, TICK_FS), ytickfont=(FIG_BOLD, TICK_FS),
          grid=true, minorgrid=false, gridwidth=1, gridcolor=:lightgray, gridalpha=0.6,
          # split: one legend per column (bottom row), centred where the curves are low
          legend=(bottom_row ? (SPLIT ? :top : :topright) : false),
          legendfont=(FIG_REGULAR, LEGEND_FS))
    if !SPLIT
        name, lux = split(panel_title, " (")   # "Low Light (150 lux)" → two lines
        annotate!(p, -18.5, common_ylims[2] * 0.88, text(name, FIG_BOLD, LABEL_FS), subplot=sp)
        annotate!(p, -18.5, common_ylims[2] * 0.78, text("(" * lux, FIG_BOLD, LABEL_FS), subplot=sp)
    end
    if bottom_row
        plot!(p, subplot=sp,
              xlabel="Time Zone Shift (Hours)",
              xguidefont=(FIG_BOLD, GUIDE_FS))
    end
end
if !SPLIT
    ymax = common_ylims[2]
    arrow_kw = (color="#A6A6A6", linewidth=6, arrow=Plots.arrow(:closed, :head), label="", subplot=1)
    plot!(p, [-11.0, -15.0], fill(0.10ymax, 2); arrow_kw...)
    annotate!(p, -8.6, 0.10ymax, text("East", FIG_BOLD, LABEL_FS - 2), subplot=1)
    plot!(p, [11.0, 15.0], fill(0.10ymax, 2); arrow_kw...)
    annotate!(p, 8.6, 0.10ymax, text("West", FIG_BOLD, LABEL_FS - 2), subplot=1)
    plot!(p, [21.5, 21.5], [0.70ymax, 0.94ymax]; arrow_kw...)
    annotate!(p, 21.5, 0.62ymax, text("Longer", FIG_BOLD, LABEL_FS - 2), subplot=1)
    annotate!(p, 21.5, 0.52ymax, text("Jet Lag", FIG_BOLD, LABEL_FS - 2), subplot=1)
end

println("   ✓ Wrapped plot created")

# ================================================================
# SAVE
# ================================================================

println("💾 Saving plot...")

output_dir = intervention_output_dir()
mkpath(output_dir)
plot_filename = joinpath(output_dir, "baseline_vs_intervention_wrapped_quartiles_steadyState$(SHOW_GENERAL ? (SPLIT ? "_withGeneral" : "_withGeneral_combined") : "").png")

savefig(p, plot_filename)
println("   ✓ Saved: $(plot_filename)")

# Per-shift numbers behind the figure (non-adapters counted as the window length; see jetlag_load_steady_state.jl)
summary_filename = joinpath(output_dir, "baseline_vs_intervention_summary_steadyState.csv")
open(summary_filename, "w") do io
    println(io, "condition,timezone_shift_h,post_shift_window_d,baseline_median_d,baseline_q25_d,baseline_q75_d,baseline_pct_not_adapted," *
                "oracle_general_best_median_d,oracle_general_best_q25_d,oracle_general_best_q75_d,oracle_general_best_pct_not_adapted," *
                "oracle_personal_best_median_d,oracle_personal_best_q25_d,oracle_personal_best_q75_d,oracle_personal_best_pct_not_adapted," *
                "oracle_general_best_start_h,oracle_general_best_duration_h,oracle_general_best_n_policies_tied,oracle_general_best_pct_helped,oracle_general_best_pct_harmed," *
                "oracle_personal_worst_median_d,oracle_personal_worst_q25_d,oracle_personal_worst_q75_d,oracle_personal_worst_pct_slowed," *
                "oracle_general_worst_median_d,oracle_general_worst_q25_d,oracle_general_worst_q75_d,oracle_general_worst_pct_slowed," *
                "oracle_general_worst_start_h,oracle_general_worst_duration_h")
    foreach(r -> println(io, r), summary_rows)
end
println("   ✓ Saved: $(summary_filename)")

println("\n🎯 Done. Adaptation times from rescuingJetLag_unified.jl (steady-state, same as 3_JetLagAnalysis).")
println("   Plot: $(plot_filename)")
println("="^70)

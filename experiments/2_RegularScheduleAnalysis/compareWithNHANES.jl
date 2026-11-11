# Compare Particle Distributions with NHANES Data
# Lara Weed
# Generate publication-ready figure comparing regular schedule particle distributions
# with NHANES self-reported sleep data.
#
# Paths from jetlag_reference.jl: particle summaries in Data/JetLag/2_RegularScheduleAnalysis/particleSummaries/,
# NHANES data from Data/NHANES/ (nhanes_sleep_file()), outputs to regular_schedule_particle_summaries_dir().
# Run with argument Combined150lux10000lux to build one figure with both light levels vs NHANES (saved under 2_RegularScheduleAnalysis/).

using CircadianPOMDP
using JLD2
using Plots
using Statistics
using StatsBase
using DataFrames
using Printf
using KernelDensity
using Dates

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

# Try to load HypothesisTests and Distributions for statistical tests
try
    using HypothesisTests
    using Distributions
    global HAS_HYPOTHESISTESTS = true
catch
    @warn "HypothesisTests or Distributions not available. Statistical tests will be skipped."
    @warn "Install with: using Pkg; Pkg.add([\"HypothesisTests\", \"Distributions\"])"
    global HAS_HYPOTHESISTESTS = false
end

# Lilliefors test for normality (composite null: mean and variance estimated from data).
# Statistic: same as one-sample KS vs Normal(μ̂, σ̂). P-value: Monte Carlo under the null.
function _lilliefors_statistic(x::AbstractVector{<:Real})
    n = length(x)
    n < 2 && return nothing
    μ, σ = mean(x), std(x)
    σ ≤ 0 && return nothing  # constant or single distinct value → not normal
    x_sorted = sort(float.(x))
    F_norm = cdf.(Normal(μ, σ), x_sorted)
    D = 0.0
    for i in 1:n
        F_emp_lo = (i - 1) / n
        F_emp_hi = i / n
        D = max(D, abs(F_emp_hi - F_norm[i]), abs(F_emp_lo - F_norm[i]))
    end
    return D
end

function lilliefors_pvalue(x::AbstractVector{<:Real}; B::Int=2000)
    D_obs = _lilliefors_statistic(x)
    D_obs === nothing && return 0.0  # reject (non-normal)
    n = length(x)
    count_ge = 0
    for _ in 1:B
        z = randn(n)
        D_b = _lilliefors_statistic(z)
        D_b !== nothing && D_b >= D_obs && (count_ge += 1)
    end
    return (count_ge + 1) / (B + 1)  # avoid p=0, standard MC p-value
end

# Try to use PyCall for reading .xpt files, fallback to CSV if needed
global HAS_PYREADSTAT = false
global pyreadstat = nothing
try
    using PyCall
    global pyreadstat = PyCall.pyimport_conda("pyreadstat", "pyreadstat")
    global HAS_PYREADSTAT = true
catch
    @warn "PyCall/pyreadstat not available. Will try to read preprocessed CSV or use alternative method."
end

# Try to load CSV package for fallback
try
    using CSV
    global HAS_CSV = true
catch
    global HAS_CSV = false
end

# Try to load DelimitedFiles for CSV fallback
try
    using DelimitedFiles
    global HAS_DELIMITEDFILES = true
catch
    global HAS_DELIMITEDFILES = false
end

# ================================================================
# HELPER FUNCTIONS
# ================================================================

"""
Convert time string (HH:MM) to decimal hours
Handles various formats including byte strings like b'21:30'
"""
function convert_time_safe(time_str)
    if ismissing(time_str) || time_str === nothing
        return NaN
    end
    
    time_str = string(time_str)
    
    # Handle byte string format: b'21:30' -> 21:30
    if startswith(time_str, "b'") && endswith(time_str, "'")
        time_str = time_str[3:end-1]  # Remove b' and '
    elseif startswith(time_str, "b\"") && endswith(time_str, "\"")
        time_str = time_str[3:end-1]  # Remove b" and "
    end
    
    if isempty(time_str) || !occursin(":", time_str)
        return NaN
    end
    
    parts = split(strip(time_str), ":")
    if length(parts) < 2
        return NaN
    end
    
    try
        h = parse(Float64, parts[1])
        m = parse(Float64, parts[2])
        return h + m / 60.0
    catch
        return NaN
    end
end

# normalize_time is now imported from CircadianPOMDP

"""
Load NHANES data from .xpt file
"""
function load_nhanes_data(xpt_path)
    if HAS_PYREADSTAT
        try
            df, meta = pyreadstat.read_xpt(xpt_path)
            @info "Successfully loaded .xpt file using pyreadstat"
            return df, meta
        catch e
            @warn "Failed to read .xpt file with pyreadstat: $e"
            @warn "Trying alternative method..."
        end
    end
    
    # Fallback: try to read as CSV if it was preprocessed
    csv_path = replace(xpt_path, ".xpt" => ".csv")
    if isfile(csv_path)
        @info "Reading preprocessed CSV file: $csv_path"
        if HAS_CSV
            return CSV.read(csv_path, DataFrame), nothing
        elseif HAS_DELIMITEDFILES
            # Try using DelimitedFiles as fallback
            data, header = readdlm(csv_path, ',', header=true)
            return DataFrame(data, vec(header)), nothing
        else
            error("Failed to read CSV file. Please install CSV.jl: using Pkg; Pkg.add(\"CSV\")")
        end
    else
        error("""
        Cannot read .xpt file. Options:
        1. Install pyreadstat: conda install -c conda-forge pyreadstat
        2. Convert .xpt to CSV first using R: library(haven); read_xpt('file.xpt') |> write_csv('file.csv')
        3. Use Python: import pandas as pd; pd.read_sas('file.xpt').to_csv('file.csv')
        """)
    end
end

"""
Process NHANES data to extract sleep timing and duration
"""
function process_nhanes_data(df)
    # Convert time columns - handle missing column names gracefully
    # Check if column exists (can be accessed as property or by name)
    bedtime_weekday = if :SLQ300 in propertynames(df) || "SLQ300" in names(df)
        converted = [convert_time_safe(x) for x in df.SLQ300]
        @info "Converted $(length(converted)) bedtime values, $(count(!isnan, converted)) valid, sample: $(converted[1:min(5, length(converted))])"
        converted
    else
        @warn "SLQ300 column not found in DataFrame. Available columns: $(names(df))"
        Float64[]
    end
    
    waketime_weekday = if :SLQ310 in propertynames(df) || "SLQ310" in names(df)
        converted = [convert_time_safe(x) for x in df.SLQ310]
        @info "Converted $(length(converted)) waketime values, $(count(!isnan, converted)) valid"
        converted
    else
        @warn "SLQ310 column not found"
        Float64[]
    end
    
    bedtime_weekend = if :SLQ320 in names(df)
        [convert_time_safe(x) for x in df.SLQ320]
    else
        Float64[]
    end
    
    waketime_weekend = if :SLQ330 in names(df)
        [convert_time_safe(x) for x in df.SLQ330]
    else
        Float64[]
    end
    
    # Normalize times to -12 to 12 range
    bedtime_weekday = normalize_time(bedtime_weekday)
    waketime_weekday = normalize_time(waketime_weekday)
    bedtime_weekend = normalize_time(bedtime_weekend)
    waketime_weekend = normalize_time(waketime_weekend)
    
    # Sleep duration (hours)
    sleep_duration_weekday = if :SLD012 in propertynames(df) || "SLD012" in names(df)
        try
            converted = []
            for x in df.SLD012
                if !ismissing(x)
                    x_str = string(x)
                    if x_str != "" && x_str != "NaN" && !startswith(x_str, "b'") && !startswith(x_str, "b\"")
                        try
                            push!(converted, parse(Float64, x_str))
                        catch
                            # Skip if can't parse
                        end
                    end
                end
            end
            @info "Converted $(length(converted)) sleep duration values"
            converted
        catch e
            @warn "Error parsing sleep duration: $e"
            Float64[]
        end
    else
        @warn "SLD012 column not found"
        Float64[]
    end
    
    sleep_duration_weekend = if :SLD013 in names(df)
        try
            [parse(Float64, string(x)) for x in df.SLD013 if !ismissing(x) && string(x) != "" && string(x) != "NaN"]
        catch
            Float64[]
        end
    else
        Float64[]
    end
    
    # Filter out NaN values
    valid_bedtime_wd = filter(!isnan, bedtime_weekday)
    valid_waketime_wd = filter(!isnan, waketime_weekday)
    valid_bedtime_we = filter(!isnan, bedtime_weekend)
    valid_waketime_we = filter(!isnan, waketime_weekend)
    
    # Calculate sleep midpoint for weekday (average of bedtime and waketime)
    # Times are already normalized to -12 to 12 range
    sleep_midpoint_weekday = []
    for i in 1:min(length(bedtime_weekday), length(waketime_weekday))
        bt = bedtime_weekday[i]
        wt = waketime_weekday[i]
        if !isnan(bt) && !isnan(wt)
            # Since times are normalized, calculate midpoint directly
            # If wake time is before bedtime (crosses midnight), add 24
            if wt < bt
                midpoint = (bt + wt + 24) / 2.0
                if midpoint > 12
                    midpoint -= 24
                end
            else
                midpoint = (bt + wt) / 2.0
            end
            push!(sleep_midpoint_weekday, midpoint)
        end
    end
    
    return (
        bedtime_weekday = valid_bedtime_wd,
        waketime_weekday = valid_waketime_wd,
        bedtime_weekend = valid_bedtime_we,
        waketime_weekend = valid_waketime_we,
        sleep_midpoint_weekday = sleep_midpoint_weekday,
        sleep_duration_weekday = sleep_duration_weekday,
        sleep_duration_weekend = sleep_duration_weekend
    )
end

# ================================================================
# MAIN SCRIPT
# ================================================================

ensure_jetlag_data_dirs()

# Name of the simulated population in legends (the forward-simulation framework is called CircadianSim in the paper)
const MODEL_LABEL = "CircadianSim"

# Optional ARGS:
#   - LowLight150lux | HighLight10000lux  → single-condition vs NHANES (default: LowLight150lux)
#   - Combined150lux10000lux              → both light levels vs NHANES on one figure (saved under 2_RegularScheduleAnalysis/)
const COMBINED_LIGHTS_ARG = "Combined150lux10000lux"
COMBINED_LIGHTS_MODE = length(ARGS) >= 1 && ARGS[1] == COMBINED_LIGHTS_ARG

if COMBINED_LIGHTS_MODE
    println("📊 $(MODEL_LABEL) 150 lux + 10 000 lux vs NHANES (combined figure)")
else
    println("📊 Comparing Particle Distributions with NHANES Data")
end
println("="^60)

normalize_time_particle(times) = begin
    times_copy = copy(times)
    times_copy[times_copy .>= 12] .-= 24
    return times_copy
end

function resolve_particle_summaries_path(condition::String)
    pth = regular_summaries_path(; condition)
    if !isfile(pth)
        pth = joinpath(regular_schedule_dir(; condition), FNAME_REGULAR_SUMMARIES)
    end
    return pth
end

function load_summaries_or_error(condition::String)
    particle_summaries_path = resolve_particle_summaries_path(condition)
    if !isfile(particle_summaries_path)
        error("Particle summaries not found for $condition. Run regularSchedule_MLE_particleSummary.jl first.\n" *
              "  Resolved: $(abspath(JETLAG_DATA_ROOT))\n  Searched: $(particle_summaries_path)")
    end
    println("   Loading from: $(abspath(particle_summaries_path))")
    s = filter_included(load_particle_summaries(particle_summaries_path))   # shared particle set (revision)
    if isempty(s)
        error("Particle summary file contains zero summaries for $condition.\n  File: $(abspath(particle_summaries_path))")
    end
    println("   ✓ Loaded $(length(s)) particle summaries ($condition)")
    return s
end

# --- Load Particle Summaries (paths from jetlag_reference.jl) ---
println("\n1. Loading particle summaries...")
if COMBINED_LIGHTS_MODE
    summaries_lo = load_summaries_or_error(REGULAR_SCHEDULE_CONDITION_LOWLIGHT)
    summaries_hi = load_summaries_or_error(REGULAR_SCHEDULE_CONDITION_HIGHLIGHT)
    summaries = summaries_lo  # placeholder for any legacy reference; combined path does not run full statistics
    particle_sleep_onsets_lo = normalize_time_particle(getproperty.(summaries_lo, :sleep_onset_mle))
    particle_sleep_offsets_lo = normalize_time_particle(getproperty.(summaries_lo, :sleep_offset_mle))
    particle_sleep_midpoints_lo = normalize_time_particle(getproperty.(summaries_lo, :sleep_midpoint_mle))
    particle_sleep_durations_lo = getproperty.(summaries_lo, :sleep_duration_mle)
    particle_cbtmin_lo = normalize_time_particle(getproperty.(summaries_lo, :CBTmin_mle))
    particle_phase_offsets_lo = getproperty.(summaries_lo, :phase_offset)
    particle_sleep_onsets_hi = normalize_time_particle(getproperty.(summaries_hi, :sleep_onset_mle))
    particle_sleep_offsets_hi = normalize_time_particle(getproperty.(summaries_hi, :sleep_offset_mle))
    particle_sleep_midpoints_hi = normalize_time_particle(getproperty.(summaries_hi, :sleep_midpoint_mle))
    particle_sleep_durations_hi = getproperty.(summaries_hi, :sleep_duration_mle)
    particle_cbtmin_hi = normalize_time_particle(getproperty.(summaries_hi, :CBTmin_mle))
    particle_phase_offsets_hi = getproperty.(summaries_hi, :phase_offset)
else
    condition = (length(ARGS) >= 1 && ARGS[1] in REGULAR_SCHEDULE_CONDITIONS) ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT
    println("Light diet condition: $condition\n")
    summaries = load_summaries_or_error(condition)
    particle_sleep_onsets = normalize_time_particle(getproperty.(summaries, :sleep_onset_mle))
    particle_sleep_offsets = normalize_time_particle(getproperty.(summaries, :sleep_offset_mle))
    particle_sleep_midpoints = normalize_time_particle(getproperty.(summaries, :sleep_midpoint_mle))
    particle_sleep_durations = getproperty.(summaries, :sleep_duration_mle)
    particle_cbtmin = normalize_time_particle(getproperty.(summaries, :CBTmin_mle))
    particle_phase_offsets = getproperty.(summaries, :phase_offset)
end

# --- Load NHANES Data ---
println("\n2. Loading NHANES data...")
nhanes_path = nhanes_sleep_file()
df_nhanes, meta = load_nhanes_data(nhanes_path)
println("   ✓ Loaded NHANES data with $(nrow(df_nhanes)) participants")

# Process NHANES data
nhanes_data = process_nhanes_data(df_nhanes)
println("   ✓ Processed NHANES sleep timing data")
println("     - Weekday bedtime: $(length(nhanes_data.bedtime_weekday)) valid entries")
println("     - Weekday waketime: $(length(nhanes_data.waketime_weekday)) valid entries")
println("     - Weekday sleep duration: $(length(nhanes_data.sleep_duration_weekday)) valid entries")

# --- Create Comparison Figure ---
println("\n3. Creating publication-ready comparison figure...")

# Native pixel size ≈ manuscript width at `dpi` so LaTeX `\includegraphics[width=12cm]{...}` does not
# shrink text further (a 3000-px-wide PNG at 300 dpi is ~25 cm wide; scaling to 12 cm makes ~30 pt look ~14 pt).
const NHANES_FIG_WIDTH_CM = 12.0
const NHANES_FIG_DPI = 300
const NHANES_FIG_WIDTH_PX = round(Int, NHANES_FIG_WIDTH_CM / 2.54 * NHANES_FIG_DPI)
# Three stacked rows (duration; onset/offset; CBTmin/phase). Panel A height = 1.5× each of B–E;
# scale total height so B–E rows keep ~the same pixel height as with the old @layout [a{0.26h}; …] split.
const NHANES_FIG_BASE_H_OVER_W = 2400 / 3000
# Old [a{0.26h}; …] gave B–E row height 0.37×H. New row fractions 3/7, 2/7, 2/7 (A = 1.5× B–E); keep that row’s pixel height via H_new*(2/7)=H_old*0.37 ⇒ scale H by 0.37*7/2.
const NHANES_FIG_HEIGHT_SCALE_A_1p5x = 0.37 * 7 / 2
# Height: aspect ratio of the manuscript figure (1304 × 1076 px as assembled)
const NHANES_FIG_HEIGHT_PX = round(Int, NHANES_FIG_WIDTH_PX * 1076 / 1304)
# Plots @layout {fh}: fraction of total height for row 1; remaining split across later rows. Use 3/7 so rows are 3:2:2 → A = 1.5× B–E.
# Font sizes (pt): tuned for ~12 cm wide multi-panel figure when saved at NHANES_FIG_DPI.
const KDE_TITLE_FS = 20
const KDE_GUIDE_FS = 22
const KDE_TICK_FS = 21
const KDE_LEGEND_FS = 18   # 8.3 pt at the SI width (6.5 in)
const KDE_PANEL_NAME_FS = 22   # bold panel name inside each panel (top left); 11.6 pt at 7.5 in wide
# Left margin: full for panels with y-axis label + tick numbers (A, B, D); tight for C/E (grid only, aligned with B/D).
const KDE_SUBPLOT_LEFT_MM = 16Plots.mm
const KDE_SUBPLOT_LEFT_MM_TIGHT = 3Plots.mm
const KDE_SUBPLOT_RIGHT_MM = 6Plots.mm
const KDE_SUBPLOT_BOTTOM_MM = 10Plots.mm
const KDE_SUBPLOT_TOP_MM = 1Plots.mm

# Publication-ready settings
default(fontfamily="Helvetica", 
        linewidth=2.5,
        markersize=6,
        dpi=NHANES_FIG_DPI)

# Define x-axis ranges for KDEs
time_range = (-12.0, 12.0)
duration_range = (0.0, 15.0)  # Sleep duration from 0 to 15 hours

# Bandwidth via Silverman's rule of thumb (KernelDensity.jl default for univariate).
# h = 1.06 * σ * n^(-1/5). Returns a default if data insufficient.
# BANDWIDTH_MULTIPLIER widens the bandwidth for smoother KDE (reduces jaggedness in NHANES).
const BANDWIDTH_MULTIPLIER = 1.5
function calculate_bandwidth(data)
    n = length(data)
    if n < 2
        return 1.0
    end
    σ = std(data)
    return BANDWIDTH_MULTIPLIER * 1.06 * σ * (n^(-1/5))
end

# Shared bandwidth for model and NHANES: use pooled sample (for statistical overlap calculations).
function shared_bandwidth(data1_float::Vector{Float64}, data2_float::Vector{Float64})
    pool = vcat(data1_float, data2_float)
    if length(pool) >= 2
        return calculate_bandwidth(pool)
    end
    if length(data1_float) >= 2
        return calculate_bandwidth(data1_float)
    end
    if length(data2_float) >= 2
        return calculate_bandwidth(data2_float)
    end
    return 1.0
end

# Compute KDE densities using KernelDensity.jl. Bandwidth is set from NHANES (data2) only,
# so it is identical across light conditions (HighLight vs LowLight) and reduces jaggedness.
# Returns (x_eval, density1_percent, density2_percent).
function compute_kde_densities(data1, data2, x_range; bandwidth=nothing)
    to_float_vec(data) = Float64[Float64(x) for x in data if x isa Number && isfinite(x)]
    data1_float = to_float_vec(data1)
    data2_float = to_float_vec(data2)
    if isempty(data1_float)
        @warn "compute_kde_densities: data1 had no finite numeric values, using zeros"
        data1_float = [0.0]
    end
    if isempty(data2_float)
        data2_float = [0.0]
    end

    x_min, x_max = x_range
    x_eval = collect(range(x_min, x_max, length=1000))

    h = bandwidth !== nothing ? bandwidth : calculate_bandwidth(data2_float)
    kde1 = kde(data1_float; bandwidth=h)
    kde2 = kde(data2_float; bandwidth=h)

    density1 = pdf(kde1, x_eval)
    density2 = pdf(kde2, x_eval)
    density1_percent = density1 .* 100
    density2_percent = density2 .* 100

    return x_eval, density1_percent, density2_percent
end

"""Bold panel name inside the panel, top left (replaces a title above the panel)."""
add_panel_name!(p, name, x_range, y_range) = isempty(name) ? p :
    annotate!(p, x_range[1] + 0.015 * (x_range[2] - x_range[1]), y_range[1] + 0.88 * (y_range[2] - y_range[1]),
              text(name, FIG_BOLD, KDE_PANEL_NAME_FS, :left))

# Helper function to create normalized KDE and plot
function plot_kde_comparison(x_eval, density1_percent, density2_percent, x_range, y_range, 
                             title_str, xlabel_str, show_legend=false, 
                             label1=MODEL_LABEL, label2="NHANES";
                             show_ylabel::Bool=true, show_ytick_labels::Bool=true)
    # Set legend position based on show_legend parameter
    legend_pos = show_legend ? :topright : false
    ylabel_str = show_ylabel ? "Percent (%)" : ""
    # Blank y tick text but keep tick positions → grid lines unchanged (panels C, E vs B, D).
    yformatter = show_ytick_labels ? :auto : (_ -> "")
    left_margin = (!show_ylabel && !show_ytick_labels) ? KDE_SUBPLOT_LEFT_MM_TIGHT : KDE_SUBPLOT_LEFT_MM
    
    # Create base plot with larger fonts for publication
    p = plot(xlims=x_range, ylims=y_range,
             title="",
             xlabel=xlabel_str,
             ylabel=ylabel_str,
             yformatter=yformatter,
             grid=true,
             gridwidth=1.0,
             gridalpha=0.15,  # Faint grid lines
             gridcolor=:black,
             legend=legend_pos,
             titlefontsize=KDE_TITLE_FS,
             titlefont=("Times", KDE_TITLE_FS, :bold),
             guidefontsize=KDE_GUIDE_FS,
             guidefont=(FIG_BOLD, KDE_GUIDE_FS),
             tickfontsize=KDE_TICK_FS,
             tickfont=(FIG_BOLD, KDE_TICK_FS),
             legendfontsize=KDE_LEGEND_FS,
             legendfont=("Times", KDE_LEGEND_FS),
             fontfamily=:serif,  # Set default font family (may help with annotations)
             bottom_margin=KDE_SUBPLOT_BOTTOM_MM,
             left_margin=left_margin,
             top_margin=KDE_SUBPLOT_TOP_MM,
             right_margin=KDE_SUBPLOT_RIGHT_MM,
             framestyle=:box)
    
    # Plot NHANES data first (solid line, grey) so model appears on top
    if !isempty(density2_percent)
        plot!(p, x_eval, density2_percent,
              linewidth=3.5,
              linestyle=:solid,
              color=:gray,
              label=label2)
    end
    
    # Plot model data second (solid line, black) so it appears on top
    plot!(p, x_eval, density1_percent,
          linewidth=4.0,
          linestyle=:solid,
          color=:black,
          label=label1)
    
    # Ensure legend text fits within box
    if show_legend
        plot!(p, legend=legend_pos,
              legendfont=("Times", KDE_LEGEND_FS),
              legendframewidth=2.0,
              legendcolumngap=18,
              legendrowgap=10,
              bgcolor=:white,
              fgcolor=:black)
    end
    
    add_panel_name!(p, title_str, x_range, y_range)
    return p
end

"""NHANES (grey) + 150 lux (black) + 10 000 lux (blue); draw order bottom→top for legend clarity."""
function plot_kde_three_way(x_eval, density_lo, density_hi, density_nhanes, x_range, y_range,
                            title_str, xlabel_str, show_legend::Bool;
                            show_ylabel::Bool=true, show_ytick_labels::Bool=true,
                            label_lo="150 lux", label_hi="10 000 lux", label_nh="NHANES")
    legend_pos = show_legend ? :topright : false
    ylabel_str = show_ylabel ? "Percent (%)" : ""
    yformatter = show_ytick_labels ? :auto : (_ -> "")
    left_margin = (!show_ylabel && !show_ytick_labels) ? KDE_SUBPLOT_LEFT_MM_TIGHT : KDE_SUBPLOT_LEFT_MM
    p = plot(xlims=x_range, ylims=y_range,
             title="", xlabel=xlabel_str, ylabel=ylabel_str, yformatter=yformatter,
             grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
             legend=legend_pos,
             titlefontsize=KDE_TITLE_FS, titlefont=("Times", KDE_TITLE_FS, :bold),
             guidefontsize=KDE_GUIDE_FS, guidefont=(FIG_BOLD, KDE_GUIDE_FS),
             tickfontsize=KDE_TICK_FS, tickfont=(FIG_BOLD, KDE_TICK_FS),
             legendfontsize=KDE_LEGEND_FS, legendfont=("Times", KDE_LEGEND_FS),
             fontfamily=:serif,
             bottom_margin=KDE_SUBPLOT_BOTTOM_MM, left_margin=left_margin,
             top_margin=KDE_SUBPLOT_TOP_MM, right_margin=KDE_SUBPLOT_RIGHT_MM,
             framestyle=:box)
    if !isempty(density_nhanes)
        plot!(p, x_eval, density_nhanes, linewidth=3.2, linestyle=:solid, color=:gray, label=label_nh)
    end
    plot!(p, x_eval, density_hi, linewidth=3.5, linestyle=:solid, color=:dodgerblue, label=label_hi)
    plot!(p, x_eval, density_lo, linewidth=4.0, linestyle=:solid, color=:black, label=label_lo)
    if show_legend
        plot!(p, legend=legend_pos, legendfont=("Times", KDE_LEGEND_FS),
              legendframewidth=2.0, legendcolumngap=12, legendrowgap=8, bgcolor=:white, fgcolor=:black)
    end
    add_panel_name!(p, title_str, x_range, y_range)
    return p
end

"""Two model curves only (CBTmin / phase); 150 lux black, 10 000 lux blue."""
function plot_kde_two_models(x_eval, density_lo, density_hi, x_range, y_range,
                             title_str, xlabel_str, show_legend::Bool;
                             show_ylabel::Bool=true, show_ytick_labels::Bool=true,
                             label_lo="150 lux", label_hi="10 000 lux")
    legend_pos = show_legend ? :topright : false
    ylabel_str = show_ylabel ? "Percent (%)" : ""
    yformatter = show_ytick_labels ? :auto : (_ -> "")
    left_margin = (!show_ylabel && !show_ytick_labels) ? KDE_SUBPLOT_LEFT_MM_TIGHT : KDE_SUBPLOT_LEFT_MM
    p = plot(xlims=x_range, ylims=y_range,
             title="", xlabel=xlabel_str, ylabel=ylabel_str, yformatter=yformatter,
             grid=true, gridwidth=1.0, gridalpha=0.15, gridcolor=:black,
             legend=legend_pos,
             titlefontsize=KDE_TITLE_FS, titlefont=("Times", KDE_TITLE_FS, :bold),
             guidefontsize=KDE_GUIDE_FS, guidefont=(FIG_BOLD, KDE_GUIDE_FS),
             tickfontsize=KDE_TICK_FS, tickfont=(FIG_BOLD, KDE_TICK_FS),
             legendfontsize=KDE_LEGEND_FS, legendfont=("Times", KDE_LEGEND_FS),
             fontfamily=:serif,
             bottom_margin=KDE_SUBPLOT_BOTTOM_MM, left_margin=left_margin,
             top_margin=KDE_SUBPLOT_TOP_MM, right_margin=KDE_SUBPLOT_RIGHT_MM,
             framestyle=:box)
    plot!(p, x_eval, density_hi, linewidth=3.5, linestyle=:solid, color=:dodgerblue, label=label_hi)
    plot!(p, x_eval, density_lo, linewidth=4.0, linestyle=:solid, color=:black, label=label_lo)
    if show_legend
        plot!(p, legend=legend_pos, legendfont=("Times", KDE_LEGEND_FS),
              legendframewidth=2.0, legendcolumngap=12, legendrowgap=8, bgcolor=:white, fgcolor=:black)
    end
    add_panel_name!(p, title_str, x_range, y_range)
    return p
end

# Normalize phase_offset to -12 to 12 range for consistency
function normalize_phase_offset(offsets)
    offsets_copy = copy(offsets)
    offsets_copy[offsets_copy .< -12] .+= 24
    offsets_copy[offsets_copy .> 12] .-= 24
    return offsets_copy
end

# Model-only KDE (CBTmin, phase offset). bandwidth_ref ensures same smoothing across light conditions.
function compute_kde_model_only(data, x_range; bandwidth_ref=nothing)
    data_float = Float64[Float64(x) for x in data if x isa Number && isfinite(x)]
    if isempty(data_float)
        data_float = [0.0]
    end
    x_min, x_max = x_range
    x_eval = collect(range(x_min, x_max, length=1000))
    h = bandwidth_ref !== nothing ? bandwidth_ref : calculate_bandwidth(data_float)
    kde_model = kde(data_float; bandwidth=h)
    density_model = pdf(kde_model, x_eval) .* 100
    return x_eval, density_model
end

# Compute reference bandwidths from NHANES only (same for HighLight and LowLight conditions).
h_time = calculate_bandwidth(Float64.(nhanes_data.bedtime_weekday))
h_duration = calculate_bandwidth(Float64.(nhanes_data.sleep_duration_weekday))

# Set y-axis range for timing plots (B, C, D, E)
y_range = (0.0, 85.0)
y_range_duration = (0.0, 40.0)

println("   Computing KDEs and determining common y-axis range...")


if COMBINED_LIGHTS_MODE
    po_lo = normalize_phase_offset(particle_phase_offsets_lo)
    po_hi = normalize_phase_offset(particle_phase_offsets_hi)
    x_eval_onset, d_lo_on, d_nh_on = compute_kde_densities(particle_sleep_onsets_lo, nhanes_data.bedtime_weekday, time_range; bandwidth=h_time)
    _, d_hi_on, _ = compute_kde_densities(particle_sleep_onsets_hi, nhanes_data.bedtime_weekday, time_range; bandwidth=h_time)
    x_eval_offset, d_lo_off, d_nh_off = compute_kde_densities(particle_sleep_offsets_lo, nhanes_data.waketime_weekday, time_range; bandwidth=h_time)
    _, d_hi_off, _ = compute_kde_densities(particle_sleep_offsets_hi, nhanes_data.waketime_weekday, time_range; bandwidth=h_time)
    x_eval_duration, d_lo_dur, d_nh_dur = compute_kde_densities(particle_sleep_durations_lo, nhanes_data.sleep_duration_weekday, duration_range; bandwidth=h_duration)
    _, d_hi_dur, _ = compute_kde_densities(particle_sleep_durations_hi, nhanes_data.sleep_duration_weekday, duration_range; bandwidth=h_duration)
    x_eval_cbtmin, d_lo_cbt = compute_kde_model_only(particle_cbtmin_lo, time_range; bandwidth_ref=h_time)
    _, d_hi_cbt = compute_kde_model_only(particle_cbtmin_hi, time_range; bandwidth_ref=h_time)
    x_eval_phase, d_lo_ph = compute_kde_model_only(po_lo, time_range; bandwidth_ref=h_time)
    _, d_hi_ph = compute_kde_model_only(po_hi, time_range; bandwidth_ref=h_time)

    p1_onset = plot_kde_three_way(x_eval_onset, d_lo_on, d_hi_on, d_nh_on,
                                  time_range, y_range, "Sleep Onset", "Time (h, relative to midnight)", true)
    p2_offset = plot_kde_three_way(x_eval_offset, d_lo_off, d_hi_off, d_nh_off,
                                   time_range, y_range, "Sleep Offset", "Time (h, relative to midnight)", true;
                                   show_ylabel=false, show_ytick_labels=false)
    p3_cbtmin = plot_kde_two_models(x_eval_cbtmin, d_lo_cbt, d_hi_cbt,
                                    time_range, y_range, display_label("CBTmin"), "Time (h, relative to midnight)", true)
    p4_duration = plot_kde_three_way(x_eval_duration, d_lo_dur, d_hi_dur, d_nh_dur,
                                       duration_range, y_range_duration, "Sleep Duration", "Duration (h)", true)
    p5_phase = plot_kde_two_models(x_eval_phase, d_lo_ph, d_hi_ph,
                                   time_range, y_range, "Phase Offset", "Hours", true;
                                   show_ylabel=false, show_ytick_labels=false)


    p_combined = plot(p4_duration, p1_onset, p2_offset, p3_cbtmin, p5_phase;
                      layout=@layout([a{0.40h}; b c; d e]),
                      size=(NHANES_FIG_WIDTH_PX, NHANES_FIG_HEIGHT_PX),
                      plot_margin=1Plots.mm, margin=2Plots.mm, bottom_margin=6Plots.mm,
                      left_margin=6Plots.mm, top_margin=3Plots.mm, right_margin=4Plots.mm)

    println("\n4. Saving combined figure...")
    output_path = joinpath(JETLAG_DATA_ROOT, DIR_REGULAR_SCHEDULE, "particle_vs_nhanes_combined_150lux_10000lux.png")
    mkpath(dirname(output_path))
    savefig(p_combined, output_path)
    println("   ✓ Saved figure to: $(output_path)")
    println("\n✅ Combined 150 lux + 10 000 lux figure complete (single-condition stats not run).")
    exit(0)
else
    particle_phase_offsets_normalized = normalize_phase_offset(particle_phase_offsets)
    x_eval_onset, density1_onset, density2_onset = compute_kde_densities(particle_sleep_onsets, nhanes_data.bedtime_weekday, time_range; bandwidth=h_time)
    x_eval_offset, density1_offset, density2_offset = compute_kde_densities(particle_sleep_offsets, nhanes_data.waketime_weekday, time_range; bandwidth=h_time)
    x_eval_duration, density1_duration, density2_duration = compute_kde_densities(particle_sleep_durations, nhanes_data.sleep_duration_weekday, duration_range; bandwidth=h_duration)
    x_eval_cbtmin, density_cbtmin = compute_kde_model_only(particle_cbtmin, time_range; bandwidth_ref=h_time)
    x_eval_phase, density_phase = compute_kde_model_only(particle_phase_offsets_normalized, time_range; bandwidth_ref=h_time)

    p1_onset = plot_kde_comparison(x_eval_onset, density1_onset, density2_onset,
                                   time_range, y_range, "Sleep Onset", "Time (h, relative to midnight)", 
                                   true, MODEL_LABEL, "NHANES")
    p2_offset = plot_kde_comparison(x_eval_offset, density1_offset, density2_offset,
                                    time_range, y_range, "Sleep Offset", "Time (h, relative to midnight)", 
                                    true, MODEL_LABEL, "NHANES"; show_ylabel=false, show_ytick_labels=false)
    p3_cbtmin = plot_kde_comparison(x_eval_cbtmin, density_cbtmin, Float64[],
                                    time_range, y_range, display_label("CBTmin"), "Time (h, relative to midnight)", 
                                    true, MODEL_LABEL, "NHANES")
    p4_duration = plot_kde_comparison(x_eval_duration, density1_duration, density2_duration,
                                      duration_range, y_range_duration, "Sleep Duration", "Duration (h)", 
                                      true, MODEL_LABEL, "NHANES")
    p5_phase = plot_kde_comparison(x_eval_phase, density_phase, Float64[],
                                   time_range, y_range, "Phase Offset", "Hours", 
                                   true, MODEL_LABEL, "NHANES"; show_ylabel=false, show_ytick_labels=false)


    p_combined = plot(p4_duration, p1_onset, p2_offset, p3_cbtmin, p5_phase;
                      layout=@layout([a{0.40h}; b c; d e]),
                      size=(NHANES_FIG_WIDTH_PX, NHANES_FIG_HEIGHT_PX),
                      plot_margin=1Plots.mm, margin=2Plots.mm, bottom_margin=6Plots.mm,
                      left_margin=6Plots.mm, top_margin=3Plots.mm, right_margin=4Plots.mm)

    println("\n4. Saving figure...")
    output_dir = regular_schedule_particle_summaries_dir(; condition)
    mkpath(output_dir)
    output_path = joinpath(output_dir, "particle_vs_nhanes_comparison.png")
    savefig(p_combined, output_path)
    println("   ✓ Saved figure to: $(output_path)")
end

# --- Statistical Analysis ---
println("\n5. Statistical Analysis:")
println("="^60)

# Print data summary before tests
println("\nData Summary:")
println("  Model particles: $(length(summaries))")
println("  NHANES participants with Sleep Onset data: $(length(nhanes_data.bedtime_weekday))")
println("  NHANES participants with Sleep Offset data: $(length(nhanes_data.waketime_weekday))")
println("  NHANES participants with Sleep Duration data: $(length(nhanes_data.sleep_duration_weekday))")

# Function to test normality (Lilliefors) and perform appropriate comparison (t-test or Mann-Whitney)
function compare_distributions(model_data, nhanes_data, metric_name)
    if isempty(model_data) || isempty(nhanes_data)
        println("\n$metric_name:")
        println("  Cannot compare: missing data")
        return
    end
    
    # Convert to Float64 arrays to ensure type compatibility
    # Filter out NaN and Inf values to ensure valid data
    model_data_clean = filter(x -> isfinite(x), Float64.(model_data))
    nhanes_data_clean = filter(x -> isfinite(x), Float64.(nhanes_data))
    
    # Report if any data was filtered out
    n_filtered_model = length(model_data) - length(model_data_clean)
    n_filtered_nhanes = length(nhanes_data) - length(nhanes_data_clean)
    if n_filtered_model > 0
        @warn "$metric_name: Filtered out $n_filtered_model non-finite values from model data"
    end
    if n_filtered_nhanes > 0
        @warn "$metric_name: Filtered out $n_filtered_nhanes non-finite values from NHANES data"
    end
    
    if isempty(model_data_clean) || isempty(nhanes_data_clean)
        println("\n$metric_name:")
        println("  Cannot compare: no valid data after filtering")
        return
    end
    
    model_data = model_data_clean
    nhanes_data = nhanes_data_clean
    
    println("\n$metric_name:")
    
    if HAS_HYPOTHESISTESTS
        # Normality: Lilliefors test only (composite null; μ, σ estimated from data; p-value by Monte Carlo).
        # For large samples, we use a subset for the test.
        model_normal = false
        nhanes_normal = false

        # For large samples, test on a random subset
        model_test_data = length(model_data) > 5000 ? sample(model_data, 5000, replace=false) : model_data
        nhanes_test_data = length(nhanes_data) > 5000 ? sample(nhanes_data, 5000, replace=false) : nhanes_data

        try
            p_model = lilliefors_pvalue(model_test_data)
            model_normal = p_model > 0.05
        catch
            model_normal = false  # use Lilliefors only; on failure treat as non-normal
        end

        try
            p_nhanes = lilliefors_pvalue(nhanes_test_data)
            nhanes_normal = p_nhanes > 0.05
        catch
            nhanes_normal = false
        end
        
        # Both normal: use t-test
        if model_normal && nhanes_normal
            try
                ttest_result = EqualVarianceTTest(model_data, nhanes_data)
                p_val = pvalue(ttest_result)
                println("  Normality test: Lilliefors (α = 0.05)")
                println("  Distribution: Normal (both groups)")
                println("  Test: Two-sample t-test")
                println("  Model:      mean=$(round(mean(model_data), digits=2)), std=$(round(std(model_data), digits=2)), n=$(length(model_data))")
                println("  NHANES:     mean=$(round(mean(nhanes_data), digits=2)), std=$(round(std(nhanes_data), digits=2)), n=$(length(nhanes_data))")
                # Format p-value appropriately
                p_str = p_val < 0.0001 ? @sprintf("%.2e", p_val) : string(round(p_val, digits=4))
                println("  p-value:    $p_str")
                
                # Effect size: Cohen's d
                pooled_std = sqrt(((length(model_data) - 1) * var(model_data) + (length(nhanes_data) - 1) * var(nhanes_data)) / (length(model_data) + length(nhanes_data) - 2))
                cohens_d = (mean(model_data) - mean(nhanes_data)) / pooled_std
                println("  Cohen's d:  $(round(cohens_d, digits=3)) (difference: $(round(mean(model_data) - mean(nhanes_data), digits=2)) h)")
            catch
                # Fallback to Mann-Whitney if t-test fails
                mw_result = MannWhitneyUTest(model_data, nhanes_data)
                p_val = pvalue(mw_result)
                println("  Normality test: Lilliefors (α = 0.05)")
                println("  Distribution: Non-normal (fallback to Mann-Whitney)")
                println("  Test: Mann-Whitney U test")
                println("  Model:      median=$(round(median(model_data), digits=2)), IQR=$(round(quantile(model_data, 0.75) - quantile(model_data, 0.25), digits=2)), n=$(length(model_data))")
                println("  NHANES:     median=$(round(median(nhanes_data), digits=2)), IQR=$(round(quantile(nhanes_data, 0.75) - quantile(nhanes_data, 0.25), digits=2)), n=$(length(nhanes_data))")
                p_str = p_val < 0.0001 ? @sprintf("%.2e", p_val) : string(round(p_val, digits=4))
                println("  p-value:    $p_str")
                
                # Effect size: Rank-biserial correlation
                n1, n2 = length(model_data), length(nhanes_data)
                U = mw_result.U
                r = 1 - (2 * U) / (n1 * n2)
                println("  Rank-biserial r: $(round(r, digits=3)) (median difference: $(round(median(model_data) - median(nhanes_data), digits=2)) h)")
            end
        else
            # Not both normal: use Mann-Whitney U test
            mw_result = MannWhitneyUTest(model_data, nhanes_data)
            p_val = pvalue(mw_result)
            println("  Normality test: Lilliefors (α = 0.05)")
            println("  Distribution: Non-normal")
            println("  Test: Mann-Whitney U test")
            println("  Model:      median=$(round(median(model_data), digits=2)), IQR=$(round(quantile(model_data, 0.75) - quantile(model_data, 0.25), digits=2)), n=$(length(model_data))")
            println("  NHANES:     median=$(round(median(nhanes_data), digits=2)), IQR=$(round(quantile(nhanes_data, 0.75) - quantile(nhanes_data, 0.25), digits=2)), n=$(length(nhanes_data))")
            p_str = p_val < 0.0001 ? @sprintf("%.2e", p_val) : string(round(p_val, digits=4))
            println("  p-value:    $p_str")
            
            # Effect size: Rank-biserial correlation (r)
            n1, n2 = length(model_data), length(nhanes_data)
            U = mw_result.U
            r = 1 - (2 * U) / (n1 * n2)
            println("  Rank-biserial r: $(round(r, digits=3)) (median difference: $(round(median(model_data) - median(nhanes_data), digits=2)) h)")
        end
        
        # Additional comparison metrics (for both normal and non-normal)
        # Kolmogorov-Smirnov test for distribution similarity
        try
            ks_test = ApproximateTwoSampleKSTest(model_data, nhanes_data)
            ks_stat = ks_test.D
            ks_p = pvalue(ks_test)
            println("  KS statistic: $(round(ks_stat, digits=4)) (max difference in CDF)")
            if ks_p < 0.0001
                println("  KS p-value:  $(@sprintf("%.2e", ks_p))")
            else
                println("  KS p-value:  $(round(ks_p, digits=4))")
            end
        catch e
            # Silently skip if KS test fails
        end
        
        # Percentile comparisons
        model_q25 = quantile(model_data, 0.25)
        model_q50 = median(model_data)
        model_q75 = quantile(model_data, 0.75)
        nhanes_q25 = quantile(nhanes_data, 0.25)
        nhanes_q50 = median(nhanes_data)
        nhanes_q75 = quantile(nhanes_data, 0.75)
        println("  Percentiles:")
        println("    25th: Model=$(round(model_q25, digits=2))h, NHANES=$(round(nhanes_q25, digits=2))h (diff: $(round(model_q25 - nhanes_q25, digits=2))h)")
        println("    50th: Model=$(round(model_q50, digits=2))h, NHANES=$(round(nhanes_q50, digits=2))h (diff: $(round(model_q50 - nhanes_q50, digits=2))h)")
        println("    75th: Model=$(round(model_q75, digits=2))h, NHANES=$(round(nhanes_q75, digits=2))h (diff: $(round(model_q75 - nhanes_q75, digits=2))h)")
        
        # Distribution overlap (using proper overlap coefficient)
        try
            model_f = Float64[Float64(x) for x in model_data if x isa Number && isfinite(x)]
            nhanes_f = Float64[Float64(x) for x in nhanes_data if x isa Number && isfinite(x)]
            if isempty(model_f); model_f = [0.0]; end
            if isempty(nhanes_f); nhanes_f = [0.0]; end
            h = shared_bandwidth(model_f, nhanes_f)
            kde_model = kde(model_f; bandwidth=h)
            kde_nhanes = kde(nhanes_f; bandwidth=h)
            # Find common range with some padding
            x_min = min(minimum(model_f), minimum(nhanes_f)) - 1.0
            x_max = max(maximum(model_f), maximum(nhanes_f)) + 1.0
            n_points = 1000
            x_eval = collect(range(x_min, x_max, length=n_points))
            dx = (x_max - x_min) / (n_points - 1)  # Step size for integration
            
            # Evaluate KDEs (pdf returns probability densities)
            pdf_model = pdf(kde_model, x_eval)
            pdf_nhanes = pdf(kde_nhanes, x_eval)
            
            # Calculate overlap coefficient: integral of min(pdf1, pdf2)
            # This gives the area under the curve of the minimum PDF at each point
            # Value ranges from 0 (no overlap) to 1 (perfect overlap)
            min_pdf = min.(pdf_model, pdf_nhanes)
            overlap = sum(min_pdf) * dx
            
            # Ensure overlap is between 0 and 1 (should be, but check for numerical issues)
            overlap = clamp(overlap, 0.0, 1.0)
            
            println("  Distribution overlap: $(round(overlap * 100, digits=1))%")
        catch e
            @warn "Failed to calculate distribution overlap: $e"
        end
    else
        # Fallback if HypothesisTests not available
        println("  Model:      mean=$(round(mean(model_data), digits=2)), std=$(round(std(model_data), digits=2)), n=$(length(model_data))")
        println("  NHANES:     mean=$(round(mean(nhanes_data), digits=2)), std=$(round(std(nhanes_data), digits=2)), n=$(length(nhanes_data))")
        println("  Note: HypothesisTests.jl not available - install with: using Pkg; Pkg.add(\"HypothesisTests\")")
    end
end

# Compare distributions for metrics with both model and NHANES data
compare_distributions(particle_sleep_onsets, nhanes_data.bedtime_weekday, "Sleep Onset (h)")
compare_distributions(particle_sleep_offsets, nhanes_data.waketime_weekday, "Sleep Offset (h)")
compare_distributions(particle_sleep_durations, nhanes_data.sleep_duration_weekday, "Sleep Duration (h)")

# --- MLE and Variance Summary ---
println("\n" * "="^60)
println("MLE (Mode from KDE) and Spread Measures Summary")
println("="^60)
println("\nNote: MLE is computed as the mode (x value at maximum KDE density),")
println("      not the sample mean. Since distributions are non-normal (see statistical tests above),")
println("      IQR (Interquartile Range) and MAD (Median Absolute Deviation)")
println("      are more robust measures of spread than standard deviation.")
println("      Standard deviation is included for reference but may be misleading")
println("      for skewed/non-normal distributions.")

# Helper function to compute MLE (mode from KDE) and spread measures
# Returns: mle, mean_val, variance, std_dev, median, q1, q3, iqr, mad, ci_lower, ci_upper, n
function compute_mle_variance(data, name)
    if isempty(data)
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 0
    end
    
    # Filter out NaN and Inf values
    data_clean = filter(x -> isfinite(x), Float64.(data))
    
    if isempty(data_clean)
        return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, 0
    end
    
    # Compute mean
    mean_val = mean(data_clean)
    
    # Compute MLE as the mode (x value at maximum KDE density)
    # Initialize with mean as fallback
    mle = mean_val
    try
        data_f = Float64[Float64(x) for x in data_clean if x isa Number && isfinite(x)]
        isempty(data_f) && (data_f = [0.0])
        h = calculate_bandwidth(data_f)
        kde_obj = kde(data_f; bandwidth=h)
        # Find range for evaluation (extend slightly beyond data range)
        x_min = minimum(data_f) - 1.0
        x_max = maximum(data_f) + 1.0
        x_eval = collect(range(x_min, x_max, length=1000))
        
        # Evaluate KDE
        density = pdf(kde_obj, x_eval)
        
        # Find x value where density is maximum (MLE = mode)
        max_idx = argmax(density)
        mle = x_eval[max_idx]
    catch e
        # Fallback to mean if KDE fails
        @warn "KDE computation failed for $name, using mean as fallback: $e"
    end
    
    variance = var(data_clean)  # Sample variance
    std_dev = sqrt(variance)  # Standard deviation
    median_val = median(data_clean)  # Median (robust central tendency)
    q1 = quantile(data_clean, 0.25)  # Q1 (25th percentile)
    q3 = quantile(data_clean, 0.75)  # Q3 (75th percentile)
    iqr = q3 - q1  # Interquartile Range (robust spread measure)
    
    # 95% Confidence Interval using percentiles (2.5th and 97.5th percentiles)
    # This is appropriate for non-normal distributions
    ci_lower = quantile(data_clean, 0.025)  # 2.5th percentile
    ci_upper = quantile(data_clean, 0.975)  # 97.5th percentile
    
    # Median Absolute Deviation (MAD) - robust measure of spread
    # MAD = median(|x_i - median(x)|)
    mad = median(abs.(data_clean .- median_val))
    n = length(data_clean)
    
    return mle, mean_val, variance, std_dev, median_val, q1, q3, iqr, mad, ci_lower, ci_upper, n
end

# Model-only metrics - compute full statistics
println("\nCBTmin (h):")
cbtmin_mle, cbtmin_mean, cbtmin_var, cbtmin_std, cbtmin_med, cbtmin_q1, cbtmin_q3, cbtmin_iqr, cbtmin_mad, cbtmin_ci_lower, cbtmin_ci_upper, cbtmin_n = compute_mle_variance(particle_cbtmin, "Model CBTmin")

if cbtmin_mle !== nothing
    println("  Model:")
    println("    MLE (Mode):     $(round(cbtmin_mle, digits=4)) h")
    println("    Mean:           $(round(cbtmin_mean, digits=4)) h")
    println("    Median:         $(round(cbtmin_med, digits=4)) h")
    println("    Q1:             $(round(cbtmin_q1, digits=4)) h")
    println("    Q3:             $(round(cbtmin_q3, digits=4)) h")
    println("    95% CI:         [$(round(cbtmin_ci_lower, digits=4)), $(round(cbtmin_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(cbtmin_std, digits=4)) h")
    println("    IQR:            $(round(cbtmin_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(cbtmin_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(cbtmin_var, digits=4)) h²")
    println("    Sample size:    $cbtmin_n")
else
    println("  Model: No valid data")
end
println("  NHANES:     N/A (not available in NHANES dataset)")

println("\nPhase Offset (h):")
phase_mle, phase_mean, phase_var, phase_std, phase_med, phase_q1, phase_q3, phase_iqr, phase_mad, phase_ci_lower, phase_ci_upper, phase_n = compute_mle_variance(particle_phase_offsets, "Model Phase Offset")

if phase_mle !== nothing
    println("  Model:")
    println("    MLE (Mode):     $(round(phase_mle, digits=4)) h")
    println("    Mean:           $(round(phase_mean, digits=4)) h")
    println("    Median:         $(round(phase_med, digits=4)) h")
    println("    Q1:             $(round(phase_q1, digits=4)) h")
    println("    Q3:             $(round(phase_q3, digits=4)) h")
    println("    95% CI:         [$(round(phase_ci_lower, digits=4)), $(round(phase_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(phase_std, digits=4)) h")
    println("    IQR:            $(round(phase_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(phase_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(phase_var, digits=4)) h²")
    println("    Sample size:    $phase_n")
else
    println("  Model: No valid data")
end
println("  NHANES:     N/A (not available in NHANES dataset)")

# Sleep Onset
println("\n📊 Sleep Onset:")
model_onset_mle, model_onset_mean, model_onset_var, model_onset_std, model_onset_med, model_onset_q1, model_onset_q3, model_onset_iqr, model_onset_mad, model_onset_ci_lower, model_onset_ci_upper, model_onset_n = compute_mle_variance(particle_sleep_onsets, "Model Sleep Onset")
nhanes_onset_mle, nhanes_onset_mean, nhanes_onset_var, nhanes_onset_std, nhanes_onset_med, nhanes_onset_q1, nhanes_onset_q3, nhanes_onset_iqr, nhanes_onset_mad, nhanes_onset_ci_lower, nhanes_onset_ci_upper, nhanes_onset_n = compute_mle_variance(nhanes_data.bedtime_weekday, "NHANES Sleep Onset")

if model_onset_mle !== nothing
    println("  Model:")
    println("    MLE (Mode):     $(round(model_onset_mle, digits=4)) h")
    println("    Mean:           $(round(model_onset_mean, digits=4)) h")
    println("    Median:         $(round(model_onset_med, digits=4)) h")
    println("    Q1:             $(round(model_onset_q1, digits=4)) h")
    println("    Q3:             $(round(model_onset_q3, digits=4)) h")
    println("    95% CI:         [$(round(model_onset_ci_lower, digits=4)), $(round(model_onset_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(model_onset_std, digits=4)) h")
    println("    IQR:            $(round(model_onset_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(model_onset_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(model_onset_var, digits=4)) h²")
    println("    Sample size:    $model_onset_n")
else
    println("  Model: No valid data")
end

if nhanes_onset_mle !== nothing
    println("  NHANES:")
    println("    MLE (Mode):     $(round(nhanes_onset_mle, digits=4)) h")
    println("    Mean:           $(round(nhanes_onset_mean, digits=4)) h")
    println("    Median:         $(round(nhanes_onset_med, digits=4)) h")
    println("    Q1:             $(round(nhanes_onset_q1, digits=4)) h")
    println("    Q3:             $(round(nhanes_onset_q3, digits=4)) h")
    println("    95% CI:         [$(round(nhanes_onset_ci_lower, digits=4)), $(round(nhanes_onset_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(nhanes_onset_std, digits=4)) h")
    println("    IQR:            $(round(nhanes_onset_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(nhanes_onset_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(nhanes_onset_var, digits=4)) h²")
    println("    Sample size:    $nhanes_onset_n")
    if model_onset_mle !== nothing
        println("  Difference:")
        println("    MLE diff:      $(round(model_onset_mle - nhanes_onset_mle, digits=4)) h")
        println("    Mean diff:     $(round(model_onset_mean - nhanes_onset_mean, digits=4)) h")
        println("    Median diff:   $(round(model_onset_med - nhanes_onset_med, digits=4)) h")
        println("    IQR diff:      $(round(model_onset_iqr - nhanes_onset_iqr, digits=4)) h")
        println("    Variance diff: $(round(model_onset_var - nhanes_onset_var, digits=4)) h²")
    end
else
    println("  NHANES: No valid data")
end

# Sleep Offset
println("\n📊 Sleep Offset:")
model_offset_mle, model_offset_mean, model_offset_var, model_offset_std, model_offset_med, model_offset_q1, model_offset_q3, model_offset_iqr, model_offset_mad, model_offset_ci_lower, model_offset_ci_upper, model_offset_n = compute_mle_variance(particle_sleep_offsets, "Model Sleep Offset")
nhanes_offset_mle, nhanes_offset_mean, nhanes_offset_var, nhanes_offset_std, nhanes_offset_med, nhanes_offset_q1, nhanes_offset_q3, nhanes_offset_iqr, nhanes_offset_mad, nhanes_offset_ci_lower, nhanes_offset_ci_upper, nhanes_offset_n = compute_mle_variance(nhanes_data.waketime_weekday, "NHANES Sleep Offset")

if model_offset_mle !== nothing
    println("  Model:")
    println("    MLE (Mode):     $(round(model_offset_mle, digits=4)) h")
    println("    Mean:           $(round(model_offset_mean, digits=4)) h")
    println("    Median:         $(round(model_offset_med, digits=4)) h")
    println("    Q1:             $(round(model_offset_q1, digits=4)) h")
    println("    Q3:             $(round(model_offset_q3, digits=4)) h")
    println("    95% CI:         [$(round(model_offset_ci_lower, digits=4)), $(round(model_offset_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(model_offset_std, digits=4)) h")
    println("    IQR:            $(round(model_offset_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(model_offset_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(model_offset_var, digits=4)) h²")
    println("    Sample size:    $model_offset_n")
else
    println("  Model: No valid data")
end

if nhanes_offset_mle !== nothing
    println("  NHANES:")
    println("    MLE (Mode):     $(round(nhanes_offset_mle, digits=4)) h")
    println("    Mean:           $(round(nhanes_offset_mean, digits=4)) h")
    println("    Median:         $(round(nhanes_offset_med, digits=4)) h")
    println("    Q1:             $(round(nhanes_offset_q1, digits=4)) h")
    println("    Q3:             $(round(nhanes_offset_q3, digits=4)) h")
    println("    95% CI:         [$(round(nhanes_offset_ci_lower, digits=4)), $(round(nhanes_offset_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(nhanes_offset_std, digits=4)) h")
    println("    IQR:            $(round(nhanes_offset_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(nhanes_offset_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(nhanes_offset_var, digits=4)) h²")
    println("    Sample size:    $nhanes_offset_n")
    if model_offset_mle !== nothing
        println("  Difference:")
        println("    MLE diff:      $(round(model_offset_mle - nhanes_offset_mle, digits=4)) h")
        println("    Mean diff:     $(round(model_offset_mean - nhanes_offset_mean, digits=4)) h")
        println("    Median diff:   $(round(model_offset_med - nhanes_offset_med, digits=4)) h")
        println("    IQR diff:      $(round(model_offset_iqr - nhanes_offset_iqr, digits=4)) h")
        println("    Variance diff: $(round(model_offset_var - nhanes_offset_var, digits=4)) h²")
    end
else
    println("  NHANES: No valid data")
end

# Helper function to compute distribution overlap for shifted data
function compute_overlap_for_shifted_data(model_data, nhanes_data, shift)
    # Shift model data and ensure Float64 for KernelDensity
    model_shifted = Float64[Float64(x) - shift for x in model_data if x isa Number && isfinite(x)]
    nhanes_f = Float64[Float64(x) for x in nhanes_data if x isa Number && isfinite(x)]
    if isempty(model_shifted); model_shifted = [0.0]; end
    if isempty(nhanes_f); nhanes_f = [0.0]; end
    h = shared_bandwidth(model_shifted, nhanes_f)
    kde_model = kde(model_shifted; bandwidth=h)
    kde_nhanes = kde(nhanes_f; bandwidth=h)
    
    # Find common range with some padding
    x_min = min(minimum(model_shifted), minimum(nhanes_f)) - 1.0
    x_max = max(maximum(model_shifted), maximum(nhanes_f)) + 1.0
    n_points = 1000
    x_eval = collect(range(x_min, x_max, length=n_points))
    dx = (x_max - x_min) / (n_points - 1)  # Step size for integration
    
    # Evaluate KDEs
    pdf_model = pdf(kde_model, x_eval)
    pdf_nhanes = pdf(kde_nhanes, x_eval)
    
    # Calculate overlap coefficient: integral of min(pdf1, pdf2)
    min_pdf = min.(pdf_model, pdf_nhanes)
    overlap = sum(min_pdf) * dx
    
    # Ensure overlap is between 0 and 1
    overlap = clamp(overlap, 0.0, 1.0)
    
    return overlap
end

# --- Peak Alignment Summary ---
println("\n" * "="^60)
println("PEAK ALIGNMENT: Timing Shifts Needed")
println("="^60)
println("\nTo align the peaks between Model and NHANES datasets:")
if model_onset_mle !== nothing && nhanes_onset_mle !== nothing
    onset_diff = model_onset_mle - nhanes_onset_mle
    println("\n📊 Sleep Onset Timing Peak:")
    println("   Model peak:     $(round(model_onset_mle, digits=4)) h")
    println("   NHANES peak:    $(round(nhanes_onset_mle, digits=4)) h")
    println("   Peak difference: $(round(onset_diff, digits=4)) h")
    if onset_diff > 0
        println("   → Shift Model EARLIER by $(round(onset_diff, digits=4)) h to align with NHANES")
        println("   → OR shift NHANES LATER by $(round(onset_diff, digits=4)) h to align with Model")
    else
        println("   → Shift Model LATER by $(round(abs(onset_diff), digits=4)) h to align with NHANES")
        println("   → OR shift NHANES EARLIER by $(round(abs(onset_diff), digits=4)) h to align with Model")
    end
end

if model_offset_mle !== nothing && nhanes_offset_mle !== nothing
    offset_diff = model_offset_mle - nhanes_offset_mle
    println("\n📊 Sleep Offset Timing Peak:")
    println("   Model peak:     $(round(model_offset_mle, digits=4)) h")
    println("   NHANES peak:    $(round(nhanes_offset_mle, digits=4)) h")
    println("   Peak difference: $(round(offset_diff, digits=4)) h")
    if offset_diff > 0
        println("   → Shift Model EARLIER by $(round(offset_diff, digits=4)) h to align with NHANES")
        println("   → OR shift NHANES LATER by $(round(offset_diff, digits=4)) h to align with Model")
    else
        println("   → Shift Model LATER by $(round(abs(offset_diff), digits=4)) h to align with NHANES")
        println("   → OR shift NHANES EARLIER by $(round(abs(offset_diff), digits=4)) h to align with Model")
    end
end

# --- Distribution Overlap for Different Alignment Approaches ---
println("\n" * "="^60)
println("DISTRIBUTION OVERLAP: Comparison of Alignment Approaches")
println("="^60)

# Sleep Onset Overlap Analysis
if model_onset_mle !== nothing && nhanes_onset_mle !== nothing && 
   model_onset_mean !== nothing && nhanes_onset_mean !== nothing &&
   model_onset_med !== nothing && nhanes_onset_med !== nothing
    println("\n📊 Sleep Onset Distribution Overlap:")
    
    # Current (unaligned) overlap - already computed in compare_distributions, but recalculate for consistency
    overlap_current_onset = compute_overlap_for_shifted_data(particle_sleep_onsets, nhanes_data.bedtime_weekday, 0.0)
    println("   Current (unaligned):     $(round(overlap_current_onset * 100, digits=1))%")
    
    # MLE alignment
    shift_mle_onset = model_onset_mle - nhanes_onset_mle
    overlap_mle_onset = compute_overlap_for_shifted_data(particle_sleep_onsets, nhanes_data.bedtime_weekday, shift_mle_onset)
    println("   MLE-aligned (shift=$(round(shift_mle_onset, digits=4))h):  $(round(overlap_mle_onset * 100, digits=1))%")
    
    # Mean alignment
    shift_mean_onset = model_onset_mean - nhanes_onset_mean
    overlap_mean_onset = compute_overlap_for_shifted_data(particle_sleep_onsets, nhanes_data.bedtime_weekday, shift_mean_onset)
    println("   Mean-aligned (shift=$(round(shift_mean_onset, digits=4))h):  $(round(overlap_mean_onset * 100, digits=1))%")
    
    # Median alignment
    shift_median_onset = model_onset_med - nhanes_onset_med
    overlap_median_onset = compute_overlap_for_shifted_data(particle_sleep_onsets, nhanes_data.bedtime_weekday, shift_median_onset)
    println("   Median-aligned (shift=$(round(shift_median_onset, digits=4))h): $(round(overlap_median_onset * 100, digits=1))%")
    
    # Find best alignment
    overlaps_onset = [("Current", overlap_current_onset), ("MLE", overlap_mle_onset), ("Mean", overlap_mean_onset), ("Median", overlap_median_onset)]
    best_onset = argmax([o[2] for o in overlaps_onset])
    println("   → Best alignment: $(overlaps_onset[best_onset][1]) ($(round(overlaps_onset[best_onset][2] * 100, digits=1))%)")
end

# Sleep Offset Overlap Analysis
if model_offset_mle !== nothing && nhanes_offset_mle !== nothing && 
   model_offset_mean !== nothing && nhanes_offset_mean !== nothing &&
   model_offset_med !== nothing && nhanes_offset_med !== nothing
    println("\n📊 Sleep Offset Distribution Overlap:")
    
    # Current (unaligned) overlap
    overlap_current_offset = compute_overlap_for_shifted_data(particle_sleep_offsets, nhanes_data.waketime_weekday, 0.0)
    println("   Current (unaligned):     $(round(overlap_current_offset * 100, digits=1))%")
    
    # MLE alignment
    shift_mle_offset = model_offset_mle - nhanes_offset_mle
    overlap_mle_offset = compute_overlap_for_shifted_data(particle_sleep_offsets, nhanes_data.waketime_weekday, shift_mle_offset)
    println("   MLE-aligned (shift=$(round(shift_mle_offset, digits=4))h):  $(round(overlap_mle_offset * 100, digits=1))%")
    
    # Mean alignment
    shift_mean_offset = model_offset_mean - nhanes_offset_mean
    overlap_mean_offset = compute_overlap_for_shifted_data(particle_sleep_offsets, nhanes_data.waketime_weekday, shift_mean_offset)
    println("   Mean-aligned (shift=$(round(shift_mean_offset, digits=4))h):  $(round(overlap_mean_offset * 100, digits=1))%")
    
    # Median alignment
    shift_median_offset = model_offset_med - nhanes_offset_med
    overlap_median_offset = compute_overlap_for_shifted_data(particle_sleep_offsets, nhanes_data.waketime_weekday, shift_median_offset)
    println("   Median-aligned (shift=$(round(shift_median_offset, digits=4))h): $(round(overlap_median_offset * 100, digits=1))%")
    
    # Find best alignment
    overlaps_offset = [("Current", overlap_current_offset), ("MLE", overlap_mle_offset), ("Mean", overlap_mean_offset), ("Median", overlap_median_offset)]
    best_offset = argmax([o[2] for o in overlaps_offset])
    println("   → Best alignment: $(overlaps_offset[best_offset][1]) ($(round(overlaps_offset[best_offset][2] * 100, digits=1))%)")
end

# Sleep Duration
println("\n📊 Sleep Duration:")
model_duration_mle, model_duration_mean, model_duration_var, model_duration_std, model_duration_med, model_duration_q1, model_duration_q3, model_duration_iqr, model_duration_mad, model_duration_ci_lower, model_duration_ci_upper, model_duration_n = compute_mle_variance(particle_sleep_durations, "Model Sleep Duration")
nhanes_duration_mle, nhanes_duration_mean, nhanes_duration_var, nhanes_duration_std, nhanes_duration_med, nhanes_duration_q1, nhanes_duration_q3, nhanes_duration_iqr, nhanes_duration_mad, nhanes_duration_ci_lower, nhanes_duration_ci_upper, nhanes_duration_n = compute_mle_variance(nhanes_data.sleep_duration_weekday, "NHANES Sleep Duration")

if model_duration_mle !== nothing
    println("  Model:")
    println("    MLE (Mode):     $(round(model_duration_mle, digits=4)) h")
    println("    Mean:           $(round(model_duration_mean, digits=4)) h")
    println("    Median:         $(round(model_duration_med, digits=4)) h")
    println("    Q1:             $(round(model_duration_q1, digits=4)) h")
    println("    Q3:             $(round(model_duration_q3, digits=4)) h")
    println("    95% CI:         [$(round(model_duration_ci_lower, digits=4)), $(round(model_duration_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(model_duration_std, digits=4)) h")
    println("    IQR:            $(round(model_duration_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(model_duration_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(model_duration_var, digits=4)) h²")
    println("    Sample size:    $model_duration_n")
else
    println("  Model: No valid data")
end

if nhanes_duration_mle !== nothing
    println("  NHANES:")
    println("    MLE (Mode):     $(round(nhanes_duration_mle, digits=4)) h")
    println("    Mean:           $(round(nhanes_duration_mean, digits=4)) h")
    println("    Median:         $(round(nhanes_duration_med, digits=4)) h")
    println("    Q1:             $(round(nhanes_duration_q1, digits=4)) h")
    println("    Q3:             $(round(nhanes_duration_q3, digits=4)) h")
    println("    95% CI:         [$(round(nhanes_duration_ci_lower, digits=4)), $(round(nhanes_duration_ci_upper, digits=4))] h")
    println("    Std Dev:        $(round(nhanes_duration_std, digits=4)) h")
    println("    IQR:            $(round(nhanes_duration_iqr, digits=4)) h (robust spread)")
    println("    MAD:            $(round(nhanes_duration_mad, digits=4)) h (robust spread)")
    println("    Variance:       $(round(nhanes_duration_var, digits=4)) h²")
    println("    Sample size:    $nhanes_duration_n")
    if model_duration_mle !== nothing
        println("  Difference:")
        println("    MLE diff:      $(round(model_duration_mle - nhanes_duration_mle, digits=4)) h")
        println("    Mean diff:     $(round(model_duration_mean - nhanes_duration_mean, digits=4)) h")
        println("    Median diff:   $(round(model_duration_med - nhanes_duration_med, digits=4)) h")
        println("    IQR diff:      $(round(model_duration_iqr - nhanes_duration_iqr, digits=4)) h")
        println("    Variance diff: $(round(model_duration_var - nhanes_duration_var, digits=4)) h²")
    end
else
    println("  NHANES: No valid data")
end

# Summary table format
println("\n" * "="^60)
println("Summary Table (Central Tendency and Spread Measures)")
println("="^60)
println("\nNote: For non-normal distributions, IQR and MAD are more robust measures of spread than Std Dev")
println("\nMetric              | Source  | MLE     | Mean    | Median  | Q1      | Q3      | 95% CI (lower) | 95% CI (upper) | IQR     | N")
println("-"^150)

if model_onset_mle !== nothing
    println(@sprintf("Sleep Onset (h)     | Model   | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    model_onset_mle, model_onset_mean, model_onset_med, model_onset_q1, model_onset_q3, 
                    model_onset_ci_lower, model_onset_ci_upper, model_onset_iqr, model_onset_n))
end
if nhanes_onset_mle !== nothing
    println(@sprintf("Sleep Onset (h)     | NHANES  | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    nhanes_onset_mle, nhanes_onset_mean, nhanes_onset_med, nhanes_onset_q1, nhanes_onset_q3,
                    nhanes_onset_ci_lower, nhanes_onset_ci_upper, nhanes_onset_iqr, nhanes_onset_n))
end

if model_offset_mle !== nothing
    println(@sprintf("Sleep Offset (h)    | Model   | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    model_offset_mle, model_offset_mean, model_offset_med, model_offset_q1, model_offset_q3,
                    model_offset_ci_lower, model_offset_ci_upper, model_offset_iqr, model_offset_n))
end
if nhanes_offset_mle !== nothing
    println(@sprintf("Sleep Offset (h)    | NHANES  | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    nhanes_offset_mle, nhanes_offset_mean, nhanes_offset_med, nhanes_offset_q1, nhanes_offset_q3,
                    nhanes_offset_ci_lower, nhanes_offset_ci_upper, nhanes_offset_iqr, nhanes_offset_n))
end

if model_duration_mle !== nothing
    println(@sprintf("Sleep Duration (h)  | Model   | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    model_duration_mle, model_duration_mean, model_duration_med, model_duration_q1, model_duration_q3,
                    model_duration_ci_lower, model_duration_ci_upper, model_duration_iqr, model_duration_n))
end
if nhanes_duration_mle !== nothing
    println(@sprintf("Sleep Duration (h)  | NHANES  | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    nhanes_duration_mle, nhanes_duration_mean, nhanes_duration_med, nhanes_duration_q1, nhanes_duration_q3,
                    nhanes_duration_ci_lower, nhanes_duration_ci_upper, nhanes_duration_iqr, nhanes_duration_n))
end

# Model-only metrics
if cbtmin_mle !== nothing
    println(@sprintf("CBTmin (h)          | Model   | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    cbtmin_mle, cbtmin_mean, cbtmin_med, cbtmin_q1, cbtmin_q3,
                    cbtmin_ci_lower, cbtmin_ci_upper, cbtmin_iqr, cbtmin_n))
end

if phase_mle !== nothing
    println(@sprintf("Phase Offset (h)    | Model   | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %13.4f | %13.4f | %7.4f | %d", 
                    phase_mle, phase_mean, phase_med, phase_q1, phase_q3,
                    phase_ci_lower, phase_ci_upper, phase_iqr, phase_n))
end

# --- Write Statistics to Files ---
println("\n6. Writing statistics to files...")
stats_text_path = joinpath(output_dir, "particle_vs_nhanes_statistics.txt")
stats_csv_path = joinpath(output_dir, "particle_vs_nhanes_statistics.csv")

# Write formatted text file
open(stats_text_path, "w") do io
    println(io, "="^80)
    println(io, "Particle Model vs NHANES: Statistical Summary")
    println(io, "="^80)
    println(io, "")
    println(io, "Generated: $(now())")
    println(io, "")
    println(io, "Note: MLE is computed as the mode (x value at maximum KDE density),")
    println(io, "      not the sample mean. Since distributions are non-normal (see statistical tests),")
    println(io, "      IQR (Interquartile Range) and MAD (Median Absolute Deviation)")
    println(io, "      are more robust measures of spread than standard deviation.")
    println(io, "")
    println(io, "="^80)
    println(io, "SLEEP ONSET")
    println(io, "="^80)
    if model_onset_mle !== nothing
        println(io, "\nModel:")
        println(io, "  MLE (Mode):     $(round(model_onset_mle, digits=4)) h")
        println(io, "  Mean:           $(round(model_onset_mean, digits=4)) h")
        println(io, "  Median:         $(round(model_onset_med, digits=4)) h")
        println(io, "  Q1:             $(round(model_onset_q1, digits=4)) h")
        println(io, "  Q3:             $(round(model_onset_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(model_onset_ci_lower, digits=4)), $(round(model_onset_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(model_onset_std, digits=4)) h")
        println(io, "  IQR:            $(round(model_onset_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(model_onset_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(model_onset_var, digits=4)) h²")
        println(io, "  Sample size:    $model_onset_n")
    end
    if nhanes_onset_mle !== nothing
        println(io, "\nNHANES:")
        println(io, "  MLE (Mode):     $(round(nhanes_onset_mle, digits=4)) h")
        println(io, "  Mean:           $(round(nhanes_onset_mean, digits=4)) h")
        println(io, "  Median:         $(round(nhanes_onset_med, digits=4)) h")
        println(io, "  Q1:             $(round(nhanes_onset_q1, digits=4)) h")
        println(io, "  Q3:             $(round(nhanes_onset_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(nhanes_onset_ci_lower, digits=4)), $(round(nhanes_onset_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(nhanes_onset_std, digits=4)) h")
        println(io, "  IQR:            $(round(nhanes_onset_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(nhanes_onset_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(nhanes_onset_var, digits=4)) h²")
        println(io, "  Sample size:    $nhanes_onset_n")
        if model_onset_mle !== nothing
            println(io, "\nDifference:")
            println(io, "  MLE diff:      $(round(model_onset_mle - nhanes_onset_mle, digits=4)) h")
            println(io, "  Mean diff:     $(round(model_onset_mean - nhanes_onset_mean, digits=4)) h")
            println(io, "  Median diff:   $(round(model_onset_med - nhanes_onset_med, digits=4)) h")
            println(io, "  IQR diff:      $(round(model_onset_iqr - nhanes_onset_iqr, digits=4)) h")
            println(io, "  Variance diff: $(round(model_onset_var - nhanes_onset_var, digits=4)) h²")
        end
    end
    
    println(io, "\n" * "="^80)
    println(io, "SLEEP OFFSET")
    println(io, "="^80)
    if model_offset_mle !== nothing
        println(io, "\nModel:")
        println(io, "  MLE (Mode):     $(round(model_offset_mle, digits=4)) h")
        println(io, "  Mean:           $(round(model_offset_mean, digits=4)) h")
        println(io, "  Median:         $(round(model_offset_med, digits=4)) h")
        println(io, "  Q1:             $(round(model_offset_q1, digits=4)) h")
        println(io, "  Q3:             $(round(model_offset_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(model_offset_ci_lower, digits=4)), $(round(model_offset_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(model_offset_std, digits=4)) h")
        println(io, "  IQR:            $(round(model_offset_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(model_offset_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(model_offset_var, digits=4)) h²")
        println(io, "  Sample size:    $model_offset_n")
    end
    if nhanes_offset_mle !== nothing
        println(io, "\nNHANES:")
        println(io, "  MLE (Mode):     $(round(nhanes_offset_mle, digits=4)) h")
        println(io, "  Mean:           $(round(nhanes_offset_mean, digits=4)) h")
        println(io, "  Median:         $(round(nhanes_offset_med, digits=4)) h")
        println(io, "  Q1:             $(round(nhanes_offset_q1, digits=4)) h")
        println(io, "  Q3:             $(round(nhanes_offset_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(nhanes_offset_ci_lower, digits=4)), $(round(nhanes_offset_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(nhanes_offset_std, digits=4)) h")
        println(io, "  IQR:            $(round(nhanes_offset_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(nhanes_offset_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(nhanes_offset_var, digits=4)) h²")
        println(io, "  Sample size:    $nhanes_offset_n")
        if model_offset_mle !== nothing
            println(io, "\nDifference:")
            println(io, "  MLE diff:      $(round(model_offset_mle - nhanes_offset_mle, digits=4)) h")
            println(io, "  Mean diff:     $(round(model_offset_mean - nhanes_offset_mean, digits=4)) h")
            println(io, "  Median diff:   $(round(model_offset_med - nhanes_offset_med, digits=4)) h")
            println(io, "  IQR diff:      $(round(model_offset_iqr - nhanes_offset_iqr, digits=4)) h")
            println(io, "  Variance diff: $(round(model_offset_var - nhanes_offset_var, digits=4)) h²")
        end
    end
    
    println(io, "\n" * "="^80)
    println(io, "SLEEP DURATION")
    println(io, "="^80)
    if model_duration_mle !== nothing
        println(io, "\nModel:")
        println(io, "  MLE (Mode):     $(round(model_duration_mle, digits=4)) h")
        println(io, "  Mean:           $(round(model_duration_mean, digits=4)) h")
        println(io, "  Median:         $(round(model_duration_med, digits=4)) h")
        println(io, "  Q1:             $(round(model_duration_q1, digits=4)) h")
        println(io, "  Q3:             $(round(model_duration_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(model_duration_ci_lower, digits=4)), $(round(model_duration_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(model_duration_std, digits=4)) h")
        println(io, "  IQR:            $(round(model_duration_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(model_duration_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(model_duration_var, digits=4)) h²")
        println(io, "  Sample size:    $model_duration_n")
    end
    if nhanes_duration_mle !== nothing
        println(io, "\nNHANES:")
        println(io, "  MLE (Mode):     $(round(nhanes_duration_mle, digits=4)) h")
        println(io, "  Mean:           $(round(nhanes_duration_mean, digits=4)) h")
        println(io, "  Median:         $(round(nhanes_duration_med, digits=4)) h")
        println(io, "  Q1:             $(round(nhanes_duration_q1, digits=4)) h")
        println(io, "  Q3:             $(round(nhanes_duration_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(nhanes_duration_ci_lower, digits=4)), $(round(nhanes_duration_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(nhanes_duration_std, digits=4)) h")
        println(io, "  IQR:            $(round(nhanes_duration_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(nhanes_duration_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(nhanes_duration_var, digits=4)) h²")
        println(io, "  Sample size:    $nhanes_duration_n")
        if model_duration_mle !== nothing
            println(io, "\nDifference:")
            println(io, "  MLE diff:      $(round(model_duration_mle - nhanes_duration_mle, digits=4)) h")
            println(io, "  Mean diff:     $(round(model_duration_mean - nhanes_duration_mean, digits=4)) h")
            println(io, "  Median diff:   $(round(model_duration_med - nhanes_duration_med, digits=4)) h")
            println(io, "  IQR diff:      $(round(model_duration_iqr - nhanes_duration_iqr, digits=4)) h")
            println(io, "  Variance diff: $(round(model_duration_var - nhanes_duration_var, digits=4)) h²")
        end
    end
    
    # Model-only metrics
    println(io, "\n" * "="^80)
    println(io, "CBTmin")
    println(io, "="^80)
    if cbtmin_mle !== nothing
        println(io, "\nModel:")
        println(io, "  MLE (Mode):     $(round(cbtmin_mle, digits=4)) h")
        println(io, "  Mean:           $(round(cbtmin_mean, digits=4)) h")
        println(io, "  Median:         $(round(cbtmin_med, digits=4)) h")
        println(io, "  Q1:             $(round(cbtmin_q1, digits=4)) h")
        println(io, "  Q3:             $(round(cbtmin_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(cbtmin_ci_lower, digits=4)), $(round(cbtmin_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(cbtmin_std, digits=4)) h")
        println(io, "  IQR:            $(round(cbtmin_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(cbtmin_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(cbtmin_var, digits=4)) h²")
        println(io, "  Sample size:    $cbtmin_n")
    end
    println(io, "\nNHANES:")
    println(io, "  N/A (not available in NHANES dataset)")
    
    println(io, "\n" * "="^80)
    println(io, "Phase Offset")
    println(io, "="^80)
    if phase_mle !== nothing
        println(io, "\nModel:")
        println(io, "  MLE (Mode):     $(round(phase_mle, digits=4)) h")
        println(io, "  Mean:           $(round(phase_mean, digits=4)) h")
        println(io, "  Median:         $(round(phase_med, digits=4)) h")
        println(io, "  Q1:             $(round(phase_q1, digits=4)) h")
        println(io, "  Q3:             $(round(phase_q3, digits=4)) h")
        println(io, "  95% CI:         [$(round(phase_ci_lower, digits=4)), $(round(phase_ci_upper, digits=4))] h")
        println(io, "  Std Dev:        $(round(phase_std, digits=4)) h")
        println(io, "  IQR:            $(round(phase_iqr, digits=4)) h (robust spread)")
        println(io, "  MAD:            $(round(phase_mad, digits=4)) h (robust spread)")
        println(io, "  Variance:       $(round(phase_var, digits=4)) h²")
        println(io, "  Sample size:    $phase_n")
    end
    println(io, "\nNHANES:")
    println(io, "  N/A (not available in NHANES dataset)")
end

# Write CSV file for easy import
open(stats_csv_path, "w") do io
    # Write header
    println(io, "Metric,Source,MLE,Mean,Median,Q1,Q3,CI_95_lower,CI_95_upper,StdDev,IQR,MAD,Variance,SampleSize")
    
    # Sleep Onset
    if model_onset_mle !== nothing
        println(io, @sprintf("Sleep Onset,Model,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            model_onset_mle, model_onset_mean, model_onset_med, model_onset_q1, model_onset_q3,
                            model_onset_ci_lower, model_onset_ci_upper, model_onset_std, 
                            model_onset_iqr, model_onset_mad, model_onset_var, model_onset_n))
    end
    if nhanes_onset_mle !== nothing
        println(io, @sprintf("Sleep Onset,NHANES,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            nhanes_onset_mle, nhanes_onset_mean, nhanes_onset_med, nhanes_onset_q1, nhanes_onset_q3,
                            nhanes_onset_ci_lower, nhanes_onset_ci_upper, nhanes_onset_std,
                            nhanes_onset_iqr, nhanes_onset_mad, nhanes_onset_var, nhanes_onset_n))
    end
    
    # Sleep Offset
    if model_offset_mle !== nothing
        println(io, @sprintf("Sleep Offset,Model,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            model_offset_mle, model_offset_mean, model_offset_med, model_offset_q1, model_offset_q3,
                            model_offset_ci_lower, model_offset_ci_upper, model_offset_std,
                            model_offset_iqr, model_offset_mad, model_offset_var, model_offset_n))
    end
    if nhanes_offset_mle !== nothing
        println(io, @sprintf("Sleep Offset,NHANES,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            nhanes_offset_mle, nhanes_offset_mean, nhanes_offset_med, nhanes_offset_q1, nhanes_offset_q3,
                            nhanes_offset_ci_lower, nhanes_offset_ci_upper, nhanes_offset_std,
                            nhanes_offset_iqr, nhanes_offset_mad, nhanes_offset_var, nhanes_offset_n))
    end
    
    # Sleep Duration
    if model_duration_mle !== nothing
        println(io, @sprintf("Sleep Duration,Model,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            model_duration_mle, model_duration_mean, model_duration_med, model_duration_q1, model_duration_q3,
                            model_duration_ci_lower, model_duration_ci_upper, model_duration_std,
                            model_duration_iqr, model_duration_mad, model_duration_var, model_duration_n))
    end
    if nhanes_duration_mle !== nothing
        println(io, @sprintf("Sleep Duration,NHANES,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            nhanes_duration_mle, nhanes_duration_mean, nhanes_duration_med, nhanes_duration_q1, nhanes_duration_q3,
                            nhanes_duration_ci_lower, nhanes_duration_ci_upper, nhanes_duration_std,
                            nhanes_duration_iqr, nhanes_duration_mad, nhanes_duration_var, nhanes_duration_n))
    end
    
    # Model-only metrics
    if cbtmin_mle !== nothing
        println(io, @sprintf("CBTmin,Model,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            cbtmin_mle, cbtmin_mean, cbtmin_med, cbtmin_q1, cbtmin_q3,
                            cbtmin_ci_lower, cbtmin_ci_upper, cbtmin_std,
                            cbtmin_iqr, cbtmin_mad, cbtmin_var, cbtmin_n))
    end
    
    if phase_mle !== nothing
        println(io, @sprintf("Phase Offset,Model,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
                            phase_mle, phase_mean, phase_med, phase_q1, phase_q3,
                            phase_ci_lower, phase_ci_upper, phase_std,
                            phase_iqr, phase_mad, phase_var, phase_n))
    end
end

println("   ✓ Saved formatted statistics to: $stats_text_path")
println("   ✓ Saved CSV statistics to: $stats_csv_path")

println("\n✅ Comparison figure generation complete!")


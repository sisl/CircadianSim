# Single figure: combined cross-correlation heatmaps (150 lux, then 10 k lux, one full-width row each)
# plus a third row with one mean |SHAP| heatmap (150 lux | 10 k lux), thick divider, single colorbar.
#
# Typography: Times; export fits FIG_TARGET_WIDTH_CM × FIG_TARGET_MAX_HEIGHT_CM at FIG_EXPORT_DPI
# (reference aspect 5200×7200 px for three stacked rows).
#
# Lara Weed

using Pkg
using CircadianPOMDP

using JLD2
using Plots
using Statistics
using StatsBase
using CSV
using DataFrames
using Printf

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

try
    using HypothesisTests
    global HAS_HYPOTHESISTESTS = true
catch
    @warn "HypothesisTests not available; p-values may be approximate."
    global HAS_HYPOTHESISTESTS = false
end

try
    using Distributions
    global HAS_DISTRIBUTIONS = true
catch
    global HAS_DISTRIBUTIONS = false
end

include(joinpath(@__DIR__, "cross_correlation_compute.jl"))
include(joinpath(@__DIR__, "cross_correlation_heatmap_library.jl"))
include(joinpath(@__DIR__, "shap_mean_abs_heatmap_lib.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

println("📊 Combined figure: correlation rows (stacked) + |SHAP| row")
println("="^70)

ensure_jetlag_data_dirs()

# Manuscript style (figure_style.jl): Greek and subscripts, bold horizontal tick labels, no titles or colorbars (the
# cells carry the values), light condition in the empty upper-right triangle, "Low Light | High Light" over the SHAP row.
const INPUT_LABELS_PLOT = display_label.(["τ", "k", "ψ", "χ", "μ_H/χ"])
const OUTPUT_LABELS_PLOT = display_label.(["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"])

const PUB_FS_GUIDE = 16
const PUB_FS_TICK = 21
const PUB_FS_PANEL = 16
const PUB_FS_LABEL = 21   # light-condition labels
# In-cell numbers (Spearman rows and |SHAP| row share one size).
const PUB_FS_CELL = 14

const FIG_TARGET_WIDTH_CM = 12.6
const FIG_TARGET_MAX_HEIGHT_CM = 18.0
const FIG_EXPORT_DPI = 300

# Reference canvas: 3 rows (corr / corr / SHAP). Taller than old 2-row layout; capped by height budget.
const REF_MEGA_W_PX = 5200
const REF_MEGA_H_PX = 7200

const W_PX_BUDGET = round(Int, FIG_TARGET_WIDTH_CM / 2.54 * FIG_EXPORT_DPI)
const H_PX_BUDGET = round(Int, FIG_TARGET_MAX_HEIGHT_CM / 2.54 * FIG_EXPORT_DPI)

const FIG_LAYOUT_SCALE = min(W_PX_BUDGET / REF_MEGA_W_PX, H_PX_BUDGET / REF_MEGA_H_PX)
const FIG_WIDTH_PX = round(Int, REF_MEGA_W_PX * FIG_LAYOUT_SCALE)
const FIG_HEIGHT_PX = round(Int, FIG_WIDTH_PX * 805 / 774)   # aspect ratio of the manuscript figure

# Full-width combined correlation heatmaps: taller native height than old side-by-side layout.
const REF_CORR_FIG_HEIGHT_PX = 2600
const CORR_FIG_HEIGHT_PX = max(400, round(Int, REF_CORR_FIG_HEIGHT_PX * FIG_LAYOUT_SCALE))
# Same width formula as `create_combined_correlation_heatmap` (n_cols=9, n_inputs=5).
const CORR_FIG_WIDTH_PX = round(Int, CORR_FIG_HEIGHT_PX * 9 / 5)

# Muted purple sequential scale for |SHAP| (see `SHAP_SEQUENTIAL_CMAP` in shap_mean_abs_heatmap_lib.jl).
const SHAP_CMAP = SHAP_SEQUENTIAL_CMAP

function load_features_for_condition(condition::String)
    particle_summaries_path = regular_summaries_path(; condition)
    if !isfile(particle_summaries_path)
        particle_summaries_path = joinpath(regular_schedule_dir(; condition), FNAME_REGULAR_SUMMARIES)
    end
    isfile(particle_summaries_path) ||
        error("Particle summaries not found for $condition. Run regularSchedule_MLE_particleSummary.jl first.\n  $particle_summaries_path")

    summaries = filter_included(load_particle_summaries(particle_summaries_path))   # shared particle set (revision)
    isempty(summaries) && error("Zero summaries in $particle_summaries_path")

    features = extract_summary_features(summaries; normalize_time_outputs = true)
    tau_values = features.inputs.tau
    k_values = features.inputs.k
    psi_values = features.inputs.psi
    chi_values = features.inputs.chi
    mu_h_over_chi_values = features.inputs.mu_h_over_chi
    s_on_values = features.outputs.s_on
    s_off_values = features.outputs.s_off
    s_dur_values = features.outputs.s_dur
    s_mp_values = features.outputs.s_mp
    cbtmin_values = features.outputs.cbtmin
    psi_obs_values = features.outputs.psi_obs

    input_data = [tau_values, k_values, psi_values, chi_values, mu_h_over_chi_values]
    keep = varying_inputs(input_data)   # ψ drops out when fixed at 0
    input_data = input_data[keep]
    output_data = [s_on_values, s_off_values, s_dur_values, s_mp_values, cbtmin_values, psi_obs_values]
    n_inputs = length(input_data)
    n_outputs = length(output_data)
    return input_data, output_data, n_inputs, n_outputs, keep
end

function combined_correlation_heatmap_for_condition(condition::String, label_lines)
    input_data, output_data, n_in, n_out, keep = load_features_for_condition(condition)
    cm, pm, ocm, opm =
        compute_spearman_correlation_matrices(input_data, output_data, n_in, n_out)
    p = create_combined_correlation_heatmap(
        cm,
        ocm,
        INPUT_LABELS_PLOT[keep],
        OUTPUT_LABELS_PLOT,
        pm,
        opm;
        param_order = restrict_order(PARAM_ORDER_COMBINED, keep),
        fig_height = CORR_FIG_HEIGHT_PX,
        ann_fontsize = PUB_FS_CELL,
        plot_title = "",
        titlefontsize = PUB_FS_PANEL,
        guidefontsize = PUB_FS_GUIDE,
        tickfontsize = PUB_FS_TICK,
        colorbar_titlefontsize = PUB_FS_GUIDE,
        left_margin = 6Plots.mm,
        right_margin = 2Plots.mm,
        top_margin = 1Plots.mm,
        bottom_margin = 3Plots.mm,
        xrotation = 0,
        framestyle = :box,
    )
    p_style!(p)
    n_cols = n_in + n_out - 1
    for (k, line) in enumerate(label_lines)   # empty upper-right triangle
        annotate!(p, n_cols - 1.0, n_out + 0.05 - 0.6 * (k - 1), text(line, FIG_BOLD, PUB_FS_LABEL))
    end
    return p
end

"""Bold tick labels, no colorbar (the cells carry the values)."""
p_style!(p) = plot!(p; colorbar=false, xtickfont=(FIG_BOLD, PUB_FS_TICK), ytickfont=(FIG_BOLD, PUB_FS_TICK))

suffix = JETLAG_ALIGNMENT_FILENAME_SUFFIX

for cond in (REGULAR_SCHEDULE_CONDITION_LOWLIGHT, REGULAR_SCHEDULE_CONDITION_HIGHLIGHT)
    p_csv = shap_matrix_csv_path(cond)
    isfile(p_csv) || error(
        "Missing SHAP matrix CSV: $p_csv\nRun: julia --project=... computeSHAP.jl $cond",
    )
end

default(
    fontfamily = "Times",
    linewidth = 1.0,
    framestyle = :box,
    dpi = FIG_EXPORT_DPI,
    foreground_color_border = :black,
    background_color = :white,
    gridalpha = 0.18,
)

p_low_comb = combined_correlation_heatmap_for_condition(
    REGULAR_SCHEDULE_CONDITION_LOWLIGHT,
    ("Low Light", "(150 lux)"),
)
p_high_comb = combined_correlation_heatmap_for_condition(
    REGULAR_SCHEDULE_CONDITION_HIGHLIGHT,
    ("High Light", "(10,000 lux)"),
)

low_in, low_out_keys, M_low = load_mean_abs_shap_matrix_csv(shap_matrix_csv_path(REGULAR_SCHEDULE_CONDITION_LOWLIGHT))
high_in, high_out_keys, M_high = load_mean_abs_shap_matrix_csv(shap_matrix_csv_path(REGULAR_SCHEDULE_CONDITION_HIGHLIGHT))
low_in == high_in && low_out_keys == high_out_keys ||
    error("SHAP CSV input/output order mismatch between conditions.")
M_low, shap_in, shap_out = shap_matrix_combined_axis_order(M_low, low_in, low_out_keys)
M_high, _, _ = shap_matrix_combined_axis_order(M_high, high_in, high_out_keys)
shap_clims = combined_shap_clims([M_low, M_high])

# Third row: one |SHAP| heatmap (150 lux | 10,000 lux), same pixel size as correlation rows, single colorbar
row_shap = mean_abs_shap_heatmap_combined_panel(
    M_low,
    M_high,
    display_label.(shap_in),
    display_label.(shap_out);
    title_str = "",
    clims = shap_clims,
    titlefontsize = PUB_FS_PANEL,
    guidefontsize = PUB_FS_GUIDE,
    tickfontsize = PUB_FS_TICK,
    ann_fontsize = PUB_FS_CELL,
    ann_hi_frac = 0.62,
    framestyle = :box,
    gridalpha = 0.22,
    colormap = SHAP_CMAP,
    colorbar_titlefontsize = PUB_FS_GUIDE,
    heatmap_size = (CORR_FIG_WIDTH_PX, CORR_FIG_HEIGHT_PX),
    left_margin = 6Plots.mm, right_margin = 2Plots.mm, top_margin = 9Plots.mm, bottom_margin = 3Plots.mm,
    xrotation = 0,
)
p_style!(row_shap)
plot!(row_shap; xlabel = "", ylabel = "")
let ni = length(shap_in), nr = length(shap_out)   # column headers above the two halves
    annotate!(row_shap, (1 + ni) / 2, nr + 0.85, text("Low Light", FIG_BOLD, PUB_FS_LABEL))
    annotate!(row_shap, ni + (1 + ni) / 2, nr + 0.85, text("High Light", FIG_BOLD, PUB_FS_LABEL))
end

mega = plot(
    p_low_comb,
    p_high_comb,
    row_shap;
    # Equal row heights so the SHAP panel matches correlation panel height (was 0.29h vs 0.355h).
    layout = @layout([a{0.333h}; b{0.333h}; c{0.334h}]),
    size = (FIG_WIDTH_PX, FIG_HEIGHT_PX),
    plot_title = "",
    background_color = :white,
)

out_path = joinpath(
    JETLAG_DATA_ROOT,
    DIR_REGULAR_SCHEDULE,
    "correlations_and_shap_combined_$(suffix)_mean.png",
)
mkpath(dirname(out_path))
savefig(mega, out_path)

w_cm = FIG_WIDTH_PX / FIG_EXPORT_DPI * 2.54
h_cm = FIG_HEIGHT_PX / FIG_EXPORT_DPI * 2.54
println(@sprintf("\n✅ Saved: %s", out_path))
println(@sprintf("   Canvas: %d × %d px at %d dpi → print size ≈ %.2f × %.2f cm (max %.1f × %.1f cm)", FIG_WIDTH_PX, FIG_HEIGHT_PX, FIG_EXPORT_DPI, w_cm, h_cm, FIG_TARGET_WIDTH_CM, FIG_TARGET_MAX_HEIGHT_CM))

# Helpers for mean |SHAP| heatmaps (used by plotCorrelationsAndSHAPCombined.jl).
# Requires jetlag_reference.jl already loaded (paths, JETLAG_ALIGNMENT_FILENAME_SUFFIX).

# Must match `PARAM_ORDER_COMBINED` / `OBS_ORDER_COMBINED` in cross_correlation_heatmap_library.jl
# so |SHAP| panels align with the combined Spearman heatmap (left block: inputs × outputs).
const COMBINED_AXIS_INPUT_PERM = [2, 1, 4, 5, 3]   # k, τ, χ, μ_H/χ, ψ
const COMBINED_AXIS_OUTPUT_PERM = [6, 5, 3, 2, 1, 4]  # ψ_obs, CBT_min, S_Dur, S_Off, S_On, S_MP

const SHAP_CSV_INPUT_ORDER = ["τ", "k", "ψ", "χ", "μ_H/χ"]
const SHAP_CSV_OUTPUT_KEYS = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "psi_obs"]

# Sequential mean |SHAP|: near-white → muted purple (works beside blue/orange Spearman heatmaps).
const SHAP_SEQUENTIAL_CMAP = cgrad(["#F5F4F8", "#E5E0EF", "#A797BC", "#564B5E"])

"""Reorder SHAP CSV matrix to the same input (column) and output (row) order as the combined correlation heatmap."""
function shap_matrix_combined_axis_order(
    M::Matrix{Float64},
    csv_input_labels::AbstractVector{<:AbstractString},
    csv_output_keys::AbstractVector{<:AbstractString},
)
    keep = BitVector([l in csv_input_labels for l in SHAP_CSV_INPUT_ORDER])   # ψ is absent when fixed at 0
    csv_input_labels == SHAP_CSV_INPUT_ORDER[keep] ||
        error("Unexpected SHAP CSV input rows.\n  expected (a subset of): $SHAP_CSV_INPUT_ORDER\n  got:      $csv_input_labels")
    csv_output_keys == SHAP_CSV_OUTPUT_KEYS ||
        error("Unexpected SHAP CSV output columns.\n  expected: $SHAP_CSV_OUTPUT_KEYS\n  got:      $csv_output_keys")
    perm = restrict_order(COMBINED_AXIS_INPUT_PERM, keep)
    Mp = M[perm, COMBINED_AXIS_OUTPUT_PERM]
    in_lbl = csv_input_labels[perm]
    out_lbl = output_key_to_label.(SHAP_CSV_OUTPUT_KEYS[COMBINED_AXIS_OUTPUT_PERM])
    return Mp, in_lbl, out_lbl
end

function shap_matrix_csv_path(condition::String)
    joinpath(
        regular_schedule_shap_dir(; condition),
        "shap_mean_abs_matrix_$(condition)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).csv",
    )
end

function load_mean_abs_shap_matrix_csv(path::String)
    df = CSV.read(path, DataFrame)
    cols = names(df)
    cols[1] == "input_feature" || error("Expected first column input_feature, got $(cols[1])")
    input_labels = String.(df[!, 1])
    output_keys = String.(cols[2:end])
    M = Matrix{Float64}(df[!, 2:end])
    size(M, 1) == length(input_labels) ||
        error("Row count mismatch in $(path)")
    return input_labels, output_keys, M
end

function output_key_to_label(key::String)
    key == "psi_obs" ? "ψ_obs" : key
end

function annotate_shap_heatmap_cells!(
    p,
    plot_matrix::AbstractMatrix{Float64},
    clims::Tuple{Float64,Float64};
    ann_fontsize::Int = 14,
    ann_hi_frac::Float64 = 0.58,
)
    lo, hi = clims
    span = hi - lo
    thresh = span > 0 ? lo + ann_hi_frac * span : lo
    for i in axes(plot_matrix, 1)
        for j in axes(plot_matrix, 2)
            v = plot_matrix[i, j]
            isfinite(v) || continue
            text_str = @sprintf("%.3f", v)
            # Dark end of sequential scale (muted purple): white ink; pale end: black
            text_color = v > thresh ? :white : :black
            annotate!(p, j, i, text(text_str, text_color, ann_fontsize, :center))
        end
    end
    return p
end

function mean_abs_shap_heatmap_panel(
    mean_abs_mat::Matrix{Float64},
    input_labels,
    output_labels;
    title_str::String,
    clims::Tuple{Float64,Float64},
    show_ylabel::Bool,
    titlefontsize::Int = 20,
    guidefontsize::Int = 14,
    tickfontsize::Int = 12,
    ann_fontsize::Int = 14,
    gridalpha::Float64 = 0.35,
    framestyle = :axes,
    colormap = SHAP_SEQUENTIAL_CMAP,
    left_margin = nothing,
    right_margin = nothing,
    top_margin = 15Plots.mm,
    bottom_margin = 20Plots.mm,
    aspect_ratio = 1.0,
    ann_hi_frac::Float64 = 0.58,
    heatmap_size::Union{Nothing,Tuple{Int,Int}} = nothing,
)
    plot_matrix = Matrix(mean_abs_mat')
    lm = left_margin === nothing ? (show_ylabel ? 22Plots.mm : 10Plots.mm) : left_margin
    rm = right_margin === nothing ? 8Plots.mm : right_margin
    # Match combined Spearman heatmap cell geometry: corr uses size (2fh, fh) for 10×6 cells →
    # cell_w/cell_h = 6/5; SHAP is 5×6 with size (fh, fh) → (fh/5)/(fh/6) = 6/5.
    ar = heatmap_size === nothing ? aspect_ratio : :auto
    base = (
        xlabel = "Model input parameters",
        ylabel = show_ylabel ? "Model output variables" : "",
        title = title_str,
        color = colormap,
        clims = clims,
        aspect_ratio = ar,
        titlefontsize = titlefontsize,
        guidefontsize = guidefontsize,
        tickfontsize = tickfontsize,
        xticks = (1:length(input_labels), input_labels),
        yticks = (1:length(output_labels), output_labels),
        colorbar = false,
        grid = true,
        gridwidth = 0.6,
        gridalpha = gridalpha,
        framestyle = framestyle,
        foreground_color_grid = RGBA(0, 0, 0, 0.18),
        left_margin = lm,
        right_margin = rm,
        top_margin = top_margin,
        bottom_margin = bottom_margin,
    )
    p = if heatmap_size === nothing
        heatmap(plot_matrix; base...)
    else
        heatmap(plot_matrix; base..., size = heatmap_size)
    end
    annotate_shap_heatmap_cells!(p, plot_matrix, clims; ann_fontsize = ann_fontsize, ann_hi_frac = ann_hi_frac)
    return p
end

"""Single-panel heatmap (same layout style as `create_combined_correlation_heatmap`): `M_low | M_high`, vertical rule at the join, one built-in colorbar — no second synthetic column (avoids GR strip + colorbar duplication)."""
function mean_abs_shap_heatmap_combined_panel(
    M_low::Matrix{Float64},
    M_high::Matrix{Float64},
    input_labels,
    output_labels;
    title_str::String = "Mean |SHAP|",
    clims::Tuple{Float64,Float64},
    divider_lw::Float64 = 5.0,
    titlefontsize::Int = 20,
    guidefontsize::Int = 14,
    tickfontsize::Int = 12,
    ann_fontsize::Int = 14,
    gridalpha::Float64 = 0.35,
    framestyle = :axes,
    colormap = SHAP_SEQUENTIAL_CMAP,
    colorbar_title::String = "Mean |SHAP|",
    colorbar_titlefontsize::Int = 14,
    left_margin = 46Plots.mm,
    right_margin = 36Plots.mm,
    top_margin = 22Plots.mm,
    bottom_margin = 40Plots.mm,
    xrotation::Real = 40,
    ann_hi_frac::Float64 = 0.58,
    heatmap_size::Tuple{Int,Int},
)
    L = Matrix(M_low')
    R = Matrix(M_high')
    size(L, 1) == size(R, 1) == length(output_labels) ||
        error("Combined SHAP: row count must match output_labels length")
    size(L, 2) == size(R, 2) == length(input_labels) ||
        error("Combined SHAP: column count must match input_labels length")
    plot_matrix = hcat(L, R)
    n_col = size(plot_matrix, 2)
    n_row = size(plot_matrix, 1)
    ni = length(input_labels)
    divider_x = ni + 0.5
    xtick_positions = vcat(1:ni, (1:ni) .+ ni)
    xtick_labels = vcat(collect(input_labels), collect(input_labels))
    p = heatmap(
        plot_matrix;
        xlabel = "Model input parameters",
        ylabel = "Model output variables",
        title = title_str,
        color = colormap,
        clims = clims,
        aspect_ratio = :auto,
        size = heatmap_size,
        titlefontsize = titlefontsize,
        guidefontsize = guidefontsize,
        tickfontsize = tickfontsize,
        xticks = (xtick_positions, xtick_labels),
        yticks = (1:length(output_labels), output_labels),
        colorbar = true,
        colorbar_title = colorbar_title,
        colorbar_titlefontsize = colorbar_titlefontsize,
        grid = true,
        gridwidth = 0.6,
        gridalpha = gridalpha,
        framestyle = framestyle,
        foreground_color_grid = RGBA(0, 0, 0, 0.18),
        left_margin = left_margin,
        right_margin = right_margin,
        top_margin = top_margin,
        bottom_margin = bottom_margin,
        xrotation = xrotation,
        xlims = (0.5, n_col + 0.5),
        ylims = (0.5, n_row + 0.5),
    )
    plot!(
        p,
        [divider_x, divider_x],
        [0.5, n_row + 0.5];
        color = :black,
        linewidth = divider_lw,
        legend = false,
    )
    annotate_shap_heatmap_cells!(p, plot_matrix, clims; ann_fontsize = ann_fontsize, ann_hi_frac = ann_hi_frac)
    return p
end

function mean_abs_shap_colorbar_column(
    clims::Tuple{Float64,Float64};
    colorbar_title::String = "Mean |SHAP|",
    colorbar_titlefontsize::Int = 14,
    top_margin = 52Plots.mm,
    bottom_margin = 48Plots.mm,
    colormap = SHAP_SEQUENTIAL_CMAP,
)
    lo, hi = clims
    # One colorbar only: a tiny heatmap + colorbar=true often draws two scales in Plots/GR.
    n = 120
    z = collect(range(lo, hi; length = n))
    x = zeros(n)
    scatter(
        x,
        z;
        marker_z = z,
        markersize = 2.5,
        markerstrokewidth = 0,
        color = colormap,
        clims = clims,
        colorbar = true,
        colorbar_title = colorbar_title,
        colorbar_titlefontsize = colorbar_titlefontsize,
        legend = false,
        framestyle = :none,
        grid = false,
        showaxis = false,
        xticks = :none,
        yticks = :none,
        xlims = (-0.15, 0.15),
        ylims = (lo, hi),
        title = "",
        left_margin = 2Plots.mm,
        right_margin = 30Plots.mm,
        top_margin = top_margin,
        bottom_margin = bottom_margin,
    )
end

function combined_shap_clims(matrices::Vector{Matrix{Float64}})
    finite_vals = Float64[]
    for M in matrices
        for v in M
            isfinite(v) && push!(finite_vals, v)
        end
    end
    isempty(finite_vals) && return (0.0, 1.0)
    vmax = maximum(finite_vals)
    vmax > 0 ? (0.0, vmax * 1.02) : (0.0, 1.0)
end

# Heatmap builders for cross-correlation analysis (same figures as computeCrossCorrelations.jl).
# Included by computeCrossCorrelations.jl and plotCorrelationsAndSHAPCombined.jl.

# Diverging Spearman ρ: blue / white / orange (Wong palette; matches jet-lag intervention curves).
const CORRELATION_DIVERGING_CMAP = cgrad(["#0072B2", "#FFFFFF", "#E69F00"])

"""Effect-size stars for a Spearman ρ: "**" for |ρ| ≥ 0.5 (strong), "*" for |ρ| ≥ 0.3 (moderate)."""
effect_size_stars(ρ) = abs(ρ) >= 0.5 ? "**" : abs(ρ) >= 0.3 ? "*" : ""

# Keep in sync with COMBINED_AXIS_INPUT_PERM / COMBINED_AXIS_OUTPUT_PERM in shap_mean_abs_heatmap_lib.jl
const PARAM_ORDER_COMBINED = [2, 1, 4, 5, 3]  # k, tau, chi, mu_h_over_chi, psi
const OBS_ORDER_COMBINED = [6, 5, 3, 2, 1, 4]  # ψ_obs, CBT_min, S_Dur, S_Off, S_On, S_MP

function create_correlation_heatmap(corr_matrix, input_labels_plot, output_labels_plot, pvalue_matrix)
    plot_matrix = corr_matrix'

    sig_matrix = similar(plot_matrix)
    for i in axes(plot_matrix, 1)
        for j in axes(plot_matrix, 2)
            if !isnan(plot_matrix[i, j])
                if !isnan(pvalue_matrix[j, i]) && pvalue_matrix[j, i] < 0.05
                    sig_matrix[i, j] = 1.0
                else
                    sig_matrix[i, j] = 0.5
                end
            else
                sig_matrix[i, j] = NaN
            end
        end
    end

    p = heatmap(plot_matrix,
                xlabel="Model Input Parameters",
                ylabel="Model Output Variables",
                title="Cross-Correlations: Inputs vs Outputs",
                color=CORRELATION_DIVERGING_CMAP,
                clims=(-1, 1),
                aspect_ratio=1.0,
                size=(1200, 1200),
                titlefontsize=20,
                guidefontsize=14,
                tickfontsize=12,
                xticks=(1:length(input_labels_plot), input_labels_plot),
                yticks=(1:length(output_labels_plot), output_labels_plot),
                colorbar_title="Spearman ρ",
                grid=true,
                gridwidth=1.0,
                gridalpha=0.3,
                left_margin=20Plots.mm,
                right_margin=15Plots.mm,
                top_margin=15Plots.mm,
                bottom_margin=20Plots.mm)

    for i in axes(plot_matrix, 1)
        for j in axes(plot_matrix, 2)
            if !isnan(plot_matrix[i, j])
                corr_val = plot_matrix[i, j]
                p_val = pvalue_matrix[j, i]

                text_str = @sprintf("%.3f", corr_val)

                text_color = abs(corr_val) > 0.5 ? :white : :black

                annotate!(p, j, i, text(text_str, text_color, 14, :center))
            end
        end
    end

    return p
end

function create_output_correlation_heatmap(corr_matrix, output_labels_plot, pvalue_matrix)
    display_matrix = copy(corr_matrix)
    for i in 1:size(corr_matrix, 1)
        display_matrix[i, i] = NaN
    end

    p = heatmap(display_matrix,
                xlabel="Model Output Variables",
                ylabel="Model Output Variables",
                title="Output-Output Correlations",
                color=CORRELATION_DIVERGING_CMAP,
                clims=(-1, 1),
                aspect_ratio=:auto,
                size=(1200, 1200),
                titlefontsize=20,
                guidefontsize=14,
                tickfontsize=12,
                xticks=(1:length(output_labels_plot), output_labels_plot),
                yticks=(1:length(output_labels_plot), output_labels_plot),
                colorbar_title="Spearman ρ",
                grid=true,
                gridwidth=1.0,
                gridalpha=0.3,
                left_margin=20Plots.mm,
                right_margin=15Plots.mm,
                top_margin=15Plots.mm,
                bottom_margin=20Plots.mm,
                xlims=(0.5, length(output_labels_plot) + 0.5),
                ylims=(0.5, length(output_labels_plot) + 0.5))

    n = length(output_labels_plot)
    for i in 1:n
        x_rect = [i-0.5, i+0.5, i+0.5, i-0.5, i-0.5]
        y_rect = [i-0.5, i-0.5, i+0.5, i+0.5, i-0.5]
        plot!(p, x_rect, y_rect,
              seriestype=:shape, fillcolor=:black, linecolor=:black, linewidth=0,
              label="", alpha=1.0, primary=false)
    end

    for i in axes(corr_matrix, 1)
        for j in axes(corr_matrix, 2)
            if i != j && !isnan(corr_matrix[i, j])
                corr_val = corr_matrix[i, j]
                p_val = pvalue_matrix[i, j]

                text_str = @sprintf("%.3f", corr_val)

                text_color = abs(corr_val) > 0.5 ? :white : :black

                annotate!(p, j, i, text(text_str, text_color, 14, :center))
            end
        end
    end

    return p
end

function create_combined_correlation_heatmap(
    input_corr_matrix,
    output_corr_matrix,
    input_labels_plot,
    output_labels_plot,
    input_pvalue_matrix,
    output_pvalue_matrix;
    fig_height::Int = 900,
    ann_fontsize::Int = 18,
    plot_title::String = "",
    titlefontsize::Int = 28,
    guidefontsize::Int = 22,
    tickfontsize::Int = 18,
    colorbar_titlefontsize::Int = 22,
    left_margin = 35Plots.mm,
    right_margin = 15Plots.mm,
    top_margin = 15Plots.mm,
    bottom_margin = 20Plots.mm,
    xrotation::Real = 0,
    framestyle = nothing,
    param_order = PARAM_ORDER_COMBINED,   # restrict_order(PARAM_ORDER_COMBINED, keep) when an input is dropped
)
    n_inputs = length(input_labels_plot)
    n_outputs = length(output_labels_plot)
    obs_order = OBS_ORDER_COMBINED
    input_corr_ordered = input_corr_matrix[param_order, :]
    input_pvalue_ordered = input_pvalue_matrix[param_order, :]
    input_labels_ordered = input_labels_plot[param_order]

    block1_corr = (input_corr_ordered')[obs_order, :]
    block1_pvalue = (input_pvalue_ordered')[obs_order, :]

    obs_order_cols = obs_order[n_outputs:-1:1]
    block2_corr = output_corr_matrix[obs_order, obs_order_cols]
    block2_pvalue = output_pvalue_matrix[obs_order, obs_order_cols]
    for i in 1:n_outputs
        for j in 1:n_outputs
            if j >= (n_outputs + 1 - i)
                block2_corr[i, j] = NaN
            end
        end
    end
    for i in 1:n_outputs
        for j in (n_outputs + 1 - i):n_outputs
            block2_pvalue[i, j] = NaN
        end
    end

    block2_corr = block2_corr[:, 1:(n_outputs-1)]
    block2_pvalue = block2_pvalue[:, 1:(n_outputs-1)]
    combined_corr = hcat(block1_corr, block2_corr)
    combined_pvalue = hcat(block1_pvalue, block2_pvalue)
    n_cols = size(combined_corr, 2)
    x_labels_block1 = input_labels_ordered
    obs_labels_y = output_labels_plot[obs_order]
    obs_labels_x = output_labels_plot[obs_order_cols[1:(n_outputs-1)]]
    xtick_positions = vcat(1:n_inputs, (1:(n_outputs-1)) .+ n_inputs)
    xtick_labels = vcat(x_labels_block1, obs_labels_x)

    fig_width = round(Int, fig_height * n_cols / n_inputs)

    display_matrix = copy(combined_corr)

    top_m = plot_title == "" ? top_margin : top_margin + 10Plots.mm

    if framestyle === nothing
        p = heatmap(
            display_matrix,
            xlabel = "",
            ylabel = "",
            title = plot_title,
            color = CORRELATION_DIVERGING_CMAP,
            clims = (-1, 1),
            aspect_ratio = :auto,
            size = (fig_width, fig_height),
            titlefontsize = titlefontsize,
            guidefontsize = guidefontsize,
            tickfontsize = tickfontsize,
            xticks = (xtick_positions, xtick_labels),
            yticks = (1:n_outputs, obs_labels_y),
            colorbar_title = "Spearman ρ",
            colorbar_titlefontsize = colorbar_titlefontsize,
            grid = false,
            left_margin = left_margin,
            right_margin = right_margin,
            top_margin = top_m,
            bottom_margin = bottom_margin,
            xlims = (0.5, n_cols + 0.5),
            ylims = (0.5, n_outputs + 0.5),
            xrotation = xrotation,
        )
    else
        p = heatmap(
            display_matrix,
            xlabel = "",
            ylabel = "",
            title = plot_title,
            color = CORRELATION_DIVERGING_CMAP,
            clims = (-1, 1),
            aspect_ratio = :auto,
            size = (fig_width, fig_height),
            titlefontsize = titlefontsize,
            guidefontsize = guidefontsize,
            tickfontsize = tickfontsize,
            xticks = (xtick_positions, xtick_labels),
            yticks = (1:n_outputs, obs_labels_y),
            colorbar_title = "Spearman ρ",
            colorbar_titlefontsize = colorbar_titlefontsize,
            grid = false,
            left_margin = left_margin,
            right_margin = right_margin,
            top_margin = top_m,
            bottom_margin = bottom_margin,
            xlims = (0.5, n_cols + 0.5),
            ylims = (0.5, n_outputs + 0.5),
            xrotation = xrotation,
            framestyle = framestyle,
        )
    end

    for i in 1:size(combined_corr, 1)
        for j in 1:size(combined_corr, 2)
            if !isnan(combined_corr[i, j])
                corr_val = combined_corr[i, j]
                # Stars mark effect size (revision, reviewer 2.6): * |ρ| ≥ 0.3 (moderate), ** |ρ| ≥ 0.5 (strong).
                # Significance (Holm-adjusted) is reported in a table, not on the figure.
                text_str = @sprintf("%.3f", corr_val) * effect_size_stars(corr_val)
                # White text only on the dark blue end; the orange end is too light for white text.
                text_color = corr_val < -0.5 ? :white : :black
                annotate!(p, j, i, text(text_str, text_color, ann_fontsize, :center))
            end
        end
    end

    return p
end

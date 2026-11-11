# Compute Cross-Correlations Between Model Inputs and Outputs
# Lara Weed
# Generate correlation matrix and LaTeX table for model parameter-output relationships.
#
# Paths from jetlag_reference.jl: particle summaries in Data/JetLag/2_RegularScheduleAnalysis/particleSummaries/,
# outputs to regular_schedule_cross_correlations_dir().
#
# This script computes Spearman correlations between:
# - Input parameters: τ (tau), k (K), ψ (phi), χ (chi), μ_H/χ (mu_h/chi)
# - Output variables: S_On, S_Off, S_Dur, S_MP, CBT_min, ψ_obs (phase_offset)

using Pkg
# Activate the CircadianPOMDP project using absolute path
script_dir = @__DIR__
using CircadianPOMDP

using JLD2
using Plots
using Statistics
using StatsBase
using DataFrames
using Printf
using Dates

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

# Try to load LaTeXStrings for text formatting in combined heatmap
try
    using LaTeXStrings
    global HAS_LATEXSTRINGS = true
catch
    @warn "LaTeXStrings not available. Bold/italic formatting will be skipped."
    global HAS_LATEXSTRINGS = false
end

# Try to load HypothesisTests for significance testing
try
    using HypothesisTests
    global HAS_HYPOTHESISTESTS = true
catch
    @warn "HypothesisTests not available. Significance testing will be skipped."
    @warn "Install with: using Pkg; Pkg.add(\"HypothesisTests\")"
    global HAS_HYPOTHESISTESTS = false
end

# Try to load Distributions for manual p-value computation
try
    using Distributions
    global HAS_DISTRIBUTIONS = true
catch
    @warn "Distributions not available. Will try to install or use alternative method."
    global HAS_DISTRIBUTIONS = false
end

include(joinpath(@__DIR__, "cross_correlation_compute.jl"))

# ================================================================
# MAIN SCRIPT
# ================================================================

println("📊 Computing Cross-Correlations Between Model Inputs and Outputs")
println("="^70)

ensure_jetlag_data_dirs()

# Optional: condition from ARGS (e.g. julia script.jl LowLight150lux). Default: HighLight10000lux.
condition = (length(ARGS) >= 1 && ARGS[1] in REGULAR_SCHEDULE_CONDITIONS) ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT
println("Light diet condition: $condition\n")

# --- Load Particle Summaries (paths from jetlag_reference.jl) ---
println("\n1. Loading particle summaries...")
particle_summaries_path = regular_summaries_path(; condition)
if !isfile(particle_summaries_path)
    particle_summaries_path = joinpath(regular_schedule_dir(; condition), FNAME_REGULAR_SUMMARIES)
end
if !isfile(particle_summaries_path)
    error("Particle summaries not found. Run regularSchedule_MLE_particleSummary.jl first.\n" *
          "  Searched: 2_RegularScheduleAnalysis/particleSummaries/ and 2_RegularScheduleAnalysis/ under $(abspath(JETLAG_DATA_ROOT))")
end
summaries = filter_included(load_particle_summaries(particle_summaries_path))   # shared particle set (revision)
if isempty(summaries)
    error("Particle summary file contains zero summaries. Run regularSchedule_MLE_particleSummary.jl first.")
end
println("   ✓ Loaded $(length(summaries)) particle summaries from $(basename(dirname(particle_summaries_path)))/")

# --- Extract Data ---
println("\n2. Extracting input parameters and output variables...")

features = extract_summary_features(summaries; normalize_time_outputs=true)
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

println("   ✓ Extracted data from $(length(tau_values)) particles")
println("   ✓ Normalized sleep onset, offset, midpoint, and CBTmin to -12 to 12 range")

# --- Compute Correlations ---
println("\n3. Computing Spearman correlations...")

# Define input and output names
input_names = ["tau", "k", "psi", "chi", "mu_h_over_chi"]
input_labels_tex = ["\$\\tau\$", "\$k\$", "\$\\psi\$", "\$\\chi\$", "\$\\frac{\\mu_\\mathrm{H}}{\\chi}\$"]
input_labels_plot = ["τ", "k", "ψ", "χ", "μ_H/χ"]  # Simpler labels for plots
input_data = [tau_values, k_values, psi_values, chi_values, mu_h_over_chi_values]
keep = varying_inputs(input_data)   # ψ drops out when fixed at 0
input_names, input_labels_tex, input_labels_plot, input_data =
    input_names[keep], input_labels_tex[keep], input_labels_plot[keep], input_data[keep]

output_names = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "psi_obs"]
output_labels_tex =["S\$" * "_\\mathrm{On}" * "\$", "S\$" * "_\\mathrm{Off}" * "\$", 
                     "S\$" * "_\\mathrm{Dur}" * "\$", "S\$" * "_\\mathrm{MP}" * "\$", 
                     "CBT\$" * "_\\mathrm{min}" * "\$", "\$\\psi_\\mathrm{obs}\$"]
output_labels_plot = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"]  # Simpler labels for plots
output_data = [s_on_values, s_off_values, s_dur_values, s_mp_values, cbtmin_values, psi_obs_values]

# Initialize correlation matrices and p-value matrices
n_inputs = length(input_names)
n_outputs = length(output_names)

correlation_matrix, pvalue_matrix, output_correlation_matrix, output_pvalue_matrix =
    compute_spearman_correlation_matrices(input_data, output_data, n_inputs, n_outputs)

println("   ✓ Computed $(n_inputs * n_outputs) input-output correlations")
println("   ✓ Computed $(n_outputs * (n_outputs - 1) ÷ 2) output-output correlations")

# --- Generate LaTeX Table ---
println("\n4. Generating LaTeX table...")

# Create output directory (from jetlag_reference.jl)
output_dir = regular_schedule_cross_correlations_dir(; condition)
mkpath(output_dir)
latex_path = joinpath(output_dir, "cross_correlations_table_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.tex")

open(latex_path, "w") do io
    println(io, "\\begin{table}[htbp]")
    println(io, "")
    println(io, "    \\centering")
    println(io, "    \\begin{tabular}{c|c|c|c|c|c|c}")
    println(io, "         \\textbf{Parameter}& \\textbf{S\$" * "_\\mathrm{On}" * "\$} & \\textbf{S\$" * "_\\mathrm{Off}" * "\$} & \\textbf{S\$" * "_\\mathrm{Dur}" * "\$}  & \\textbf{S\$" * "_\\mathrm{MP}" * "\$} & \\textbf{CBT\$" * "_\\mathrm{min}" * "\$} & \\textbf{\$\\psi_\\mathrm{obs}\$}  \\\\")
    println(io, "         \\hline")
    
    # Output rows (diagonal is 1.0)
    # Format needed: S$_\mathrm{On}$ where $...$ is math mode
    # In Julia strings: \$ writes literal $, \\ writes literal \
    # So "S\$_\$\\mathrm{On}\$" writes: S + $ + _ + $ + \mathrm{On} + $
    # But we need: S + $ + _\mathrm{On} + $ (underscore inside math mode)
    # The correct format: "S\$" + "_" + "\\mathrm{On}" + "\$"
    # Actually, the underscore should be inside the math delimiters
    # So: "S" + "\$" + "_\\mathrm{On}" + "\$" = "S\$" * "_\\mathrm{On}" * "\$"
    output_labels_tex_table = output_labels_tex
    for (j, output_label) in enumerate(output_labels_tex_table)
        print(io, "         $output_label")
        for k in 1:n_outputs
            if j == k
                print(io, "& 1.0 ")
            else
                corr_val = output_correlation_matrix[j, k]
                p_val = output_pvalue_matrix[j, k]
                
                if isnan(corr_val)
                    print(io, "& ")
                else
                    # Format correlation value
                    corr_str = @sprintf("%.3f", corr_val)
                    
                    # Bold if p < 0.01 (strongest significance)
                    # Italic if 0.01 <= p < 0.05 (moderate significance)
                    if !isnan(p_val)
                        if p_val < 0.01
                            corr_str = "\\textbf{$corr_str}"
                        elseif p_val < 0.05
                            corr_str = "\\textit{$corr_str}"
                        end
                    end
                    
                    print(io, "& $corr_str ")
                end
            end
        end
        println(io, "\\\\")
    end
    println(io, "         ")
    
    # Input rows
    for (i, input_label) in enumerate(input_labels_tex)
        print(io, "         $input_label")
        for j in 1:n_outputs
            corr_val = correlation_matrix[i, j]
            p_val = pvalue_matrix[i, j]
            
            if isnan(corr_val)
                print(io, "& ")
            else
                # Format correlation value
                corr_str = @sprintf("%.3f", corr_val)
                
                # Bold if p < 0.01 (strongest significance)
                # Italic if 0.01 <= p < 0.05 (moderate significance)
                if !isnan(p_val)
                    if p_val < 0.01
                        corr_str = "\\textbf{$corr_str}"
                    elseif p_val < 0.05
                        corr_str = "\\textit{$corr_str}"
                    end
                end
                
                print(io, "& $corr_str ")
            end
        end
        println(io, "\\\\")
    end
    
    println(io, "    \\end{tabular}")
    println(io, "    \\caption{Cross-correlations (Spearman) between model inputs and outputs. Bolded values have p\$<0.01\$, italicized values have 0.01\$\\leq\$p\$<0.05\$.}")
    println(io, "    \\label{tab:Reg Sched Correlations}")
    println(io, "\\end{table}")
end

println("   ✓ Saved LaTeX table to: $latex_path")

# --- Generate Correlation Plots ---
println("\n5. Generating correlation plots...")

# Publication-ready settings
default(fontfamily="Helvetica", 
        linewidth=2.5,
        markersize=6,
        dpi=300)

# Enable LaTeX rendering if available
if HAS_LATEXSTRINGS
    println("   Debug: LaTeXStrings loaded - LaTeX formatting will be used for combined heatmap")
else
    println("   Debug: LaTeXStrings not available - installing...")
    try
        using Pkg
        Pkg.add("LaTeXStrings")
        using LaTeXStrings
        global HAS_LATEXSTRINGS = true
        println("   Debug: LaTeXStrings installed and loaded")
    catch e
        @warn "Could not install/load LaTeXStrings: $e"
        global HAS_LATEXSTRINGS = false
    end
end

include(joinpath(@__DIR__, "cross_correlation_heatmap_library.jl"))

# Create heatmap for input-output correlations
heatmap_plot = create_correlation_heatmap(correlation_matrix, input_labels_plot, output_labels_plot, pvalue_matrix)
heatmap_path = joinpath(output_dir, "cross_correlations_heatmap_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png")
savefig(heatmap_plot, heatmap_path)
println("   ✓ Saved input-output correlation heatmap to: $heatmap_path")

output_heatmap_plot = create_output_correlation_heatmap(output_correlation_matrix, output_labels_plot, output_pvalue_matrix)
output_heatmap_path = joinpath(output_dir, "output_correlations_heatmap_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png")
savefig(output_heatmap_plot, output_heatmap_path)
println("   ✓ Saved output-output correlation heatmap to: $output_heatmap_path")

combined_heatmap_plot = create_combined_correlation_heatmap(
    correlation_matrix, output_correlation_matrix,
    input_labels_plot, output_labels_plot,
    pvalue_matrix, output_pvalue_matrix; param_order=restrict_order(PARAM_ORDER_COMBINED, keep))
combined_heatmap_path = joinpath(output_dir, "combined_correlations_heatmap_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png")
savefig(combined_heatmap_plot, combined_heatmap_path)
println("   ✓ Saved combined correlation heatmap to: $combined_heatmap_path")

# Create individual scatter plots for significant correlations
function create_scatter_plots(input_data, input_names, input_labels_plot, 
                              output_data, output_names, output_labels_plot,
                              corr_matrix, pvalue_matrix)
    plots_list = []
    
    for i in 1:length(input_names)
        for j in 1:length(output_names)
            corr_val = corr_matrix[i, j]
            p_val = pvalue_matrix[i, j]
            
            # Only plot if correlation exists and is significant
            if !isnan(corr_val) && !isnan(p_val) && p_val < 0.05
                input_vec = input_data[i]
                output_vec = output_data[j]
                
                # Filter valid data
                valid_indices = Int[]
                for idx in eachindex(input_vec)
                    if idx <= length(output_vec) && isfinite(input_vec[idx]) && isfinite(output_vec[idx])
                        push!(valid_indices, idx)
                    end
                end
                input_clean = [input_vec[idx] for idx in valid_indices]
                output_clean = [output_vec[idx] for idx in valid_indices]
                
                # Create scatter plot
                p = scatter(input_clean, output_clean,
                           xlabel=input_labels_plot[i],
                           ylabel=output_labels_plot[j],
                           title=@sprintf("%s vs %s\nρ=%.3f, p=%.4f", 
                                       input_labels_plot[i], output_labels_plot[j], corr_val, p_val),
                           alpha=0.5,
                           markersize=3,
                           titlefontsize=50,
                           guidefontsize=40,
                           tickfontsize=40,
                           grid=true,
                           gridwidth=1.0,
                           gridalpha=0.3)
                
                # Add regression line
                if length(input_clean) >= 2
                    try
                        # Simple linear regression
                        coeffs = [ones(length(input_clean)) input_clean] \ output_clean
                        x_line = [minimum(input_clean), maximum(input_clean)]
                        y_line = coeffs[1] .+ coeffs[2] .* x_line
                        plot!(p, x_line, y_line, 
                             linewidth=2, 
                             linestyle=:dash, 
                             color=:red,
                             label="")
                    catch
                        # Skip regression line if it fails
                    end
                end
                
                push!(plots_list, p)
            end
        end
    end
    
    return plots_list
end

# Create scatter plots for significant input-output correlations
scatter_plots = create_scatter_plots(input_data, input_names, input_labels_plot,
                                     output_data, output_names, output_labels_plot,
                                     correlation_matrix, pvalue_matrix)

# Create scatter plots for significant output-output correlations
function create_output_scatter_plots(output_data, output_names, output_labels_plot,
                                     corr_matrix, pvalue_matrix)
    plots_list = []
    
    for i in 1:length(output_names)
        for j in (i+1):length(output_names)  # Only upper triangle to avoid duplicates
            corr_val = corr_matrix[i, j]
            p_val = pvalue_matrix[i, j]
            
            # Only plot if correlation exists and is significant
            if !isnan(corr_val) && !isnan(p_val) && p_val < 0.05
                output_vec1 = output_data[i]
                output_vec2 = output_data[j]
                
                # Filter valid data
                valid_indices = Int[]
                for idx in eachindex(output_vec1)
                    if idx <= length(output_vec2) && isfinite(output_vec1[idx]) && isfinite(output_vec2[idx])
                        push!(valid_indices, idx)
                    end
                end
                output_clean1 = [output_vec1[idx] for idx in valid_indices]
                output_clean2 = [output_vec2[idx] for idx in valid_indices]
                
                # Create scatter plot
                p = scatter(output_clean1, output_clean2,
                           xlabel=output_labels_plot[i],
                           ylabel=output_labels_plot[j],
                           title=@sprintf("%s vs %s\nρ=%.3f, p=%.4f", 
                                       output_labels_plot[i], output_labels_plot[j], corr_val, p_val),
                           alpha=0.5,
                           markersize=3,
                           titlefontsize=50,
                           guidefontsize=40,
                           tickfontsize=40,
                           grid=true,
                           gridwidth=1.0,
                           gridalpha=0.3)
                
                # Add regression line
                if length(output_clean1) >= 2
                    try
                        # Simple linear regression
                        coeffs = [ones(length(output_clean1)) output_clean1] \ output_clean2
                        x_line = [minimum(output_clean1), maximum(output_clean1)]
                        y_line = coeffs[1] .+ coeffs[2] .* x_line
                        plot!(p, x_line, y_line, 
                             linewidth=2, 
                             linestyle=:dash, 
                             color=:red,
                             label="")
                    catch
                        # Skip regression line if it fails
                    end
                end
                
                push!(plots_list, p)
            end
        end
    end
    
    return plots_list
end

output_scatter_plots = create_output_scatter_plots(output_data, output_names, output_labels_plot,
                                                   output_correlation_matrix, output_pvalue_matrix)

# Combine all scatter plots
all_scatter_plots = vcat(scatter_plots, output_scatter_plots)

if !isempty(all_scatter_plots)
    # Arrange scatter plots in a grid
    n_plots = length(all_scatter_plots)
    n_cols = min(3, n_plots)
    n_rows = ceil(Int, n_plots / n_cols)
    
    scatter_grid = plot(all_scatter_plots..., 
                       layout=(n_rows, n_cols),
                       size=(1800, 600 * n_rows),
                       margin=10Plots.mm)
    
    scatter_path = joinpath(output_dir, "cross_correlations_scatter_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png")
    savefig(scatter_grid, scatter_path)
    println("   ✓ Saved $(length(scatter_plots)) input-output and $(length(output_scatter_plots)) output-output scatter plots to: $scatter_path")
else
    println("   ⚠ No significant correlations found (p < 0.05) for scatter plots")
end

# --- Print Summary Statistics ---
println("\n6. Summary Statistics:")
println("="^70)

println("\nInput Parameters:")
println("  τ (tau):        mean=$(round(mean(tau_values), digits=4)), std=$(round(std(tau_values), digits=4))")
println("  k (K):          mean=$(round(mean(k_values), digits=4)), std=$(round(std(k_values), digits=4))")
println("  ψ (phi):        mean=$(round(mean(psi_values), digits=4)), std=$(round(std(psi_values), digits=4))")
println("  χ (chi):        mean=$(round(mean(chi_values), digits=4)), std=$(round(std(chi_values), digits=4))")
println("  μ_H/χ:          mean=$(round(mean(mu_h_over_chi_values), digits=4)), std=$(round(std(mu_h_over_chi_values), digits=4))")

println("\nOutput Variables:")
println("  S_On:           mean=$(round(mean(s_on_values), digits=4)), std=$(round(std(s_on_values), digits=4))")
println("  S_Off:          mean=$(round(mean(s_off_values), digits=4)), std=$(round(std(s_off_values), digits=4))")
println("  S_Dur:          mean=$(round(mean(s_dur_values), digits=4)), std=$(round(std(s_dur_values), digits=4))")
println("  S_MP:           mean=$(round(mean(s_mp_values), digits=4)), std=$(round(std(s_mp_values), digits=4))")
println("  CBT_min:        mean=$(round(mean(cbtmin_values), digits=4)), std=$(round(std(cbtmin_values), digits=4))")
println("  ψ_obs:          mean=$(round(mean(psi_obs_values), digits=4)), std=$(round(std(psi_obs_values), digits=4))")

# Count significant correlations
n_significant_input_output = count(x -> !isnan(x) && x < 0.05, pvalue_matrix)
n_significant_output_output = count(i -> i[1] != i[2] && !isnan(output_pvalue_matrix[i[1], i[2]]) && output_pvalue_matrix[i[1], i[2]] < 0.05, 
                                   [(i, j) for i in 1:n_outputs, j in 1:n_outputs])
println("\nSignificant Correlations (p < 0.05):")
println("  Input-Output: $n_significant_input_output out of $(n_inputs * n_outputs)")
println("  Output-Output: $n_significant_output_output out of $(n_outputs * (n_outputs - 1) ÷ 2)")

# Print top correlations (including both input-output and output-output)
println("\nTop Correlations (by absolute value):")
all_corrs = []
# Input-output correlations
for i in 1:n_inputs
    for j in 1:n_outputs
        if !isnan(correlation_matrix[i, j])
            push!(all_corrs, (abs(correlation_matrix[i, j]), 
                             correlation_matrix[i, j],
                             input_labels_tex[i], 
                             output_labels_tex[j],
                             pvalue_matrix[i, j],
                             "input-output"))
        end
    end
end
# Output-output correlations
for i in 1:n_outputs
    for j in (i+1):n_outputs  # Only upper triangle to avoid duplicates
        if !isnan(output_correlation_matrix[i, j])
            push!(all_corrs, (abs(output_correlation_matrix[i, j]), 
                             output_correlation_matrix[i, j],
                             output_labels_tex[i], 
                             output_labels_tex[j],
                             output_pvalue_matrix[i, j],
                             "output-output"))
        end
    end
end

sort!(all_corrs, by=x -> x[1], rev=true)
for (idx, (_, corr_val, var1_lab, var2_lab, p_val, corr_type)) in enumerate(all_corrs[1:min(15, length(all_corrs))])
    sig_str = !isnan(p_val) && p_val < 0.05 ? " *" : ""
    # Use plot labels for console output (simpler)
    # Find var1
    var1_idx = findfirst(x -> x == var1_lab, input_labels_tex)
    if var1_idx !== nothing
        var1_display = input_labels_plot[var1_idx]
    else
        var1_idx = findfirst(x -> x == var1_lab, output_labels_tex)
        var1_display = var1_idx !== nothing ? output_labels_plot[var1_idx] : var1_lab
    end
    
    # Find var2
    var2_idx = findfirst(x -> x == var2_lab, output_labels_tex)
    var2_display = var2_idx !== nothing ? output_labels_plot[var2_idx] : var2_lab
    
    type_str = corr_type == "input-output" ? "[I-O]" : "[O-O]"
    println("  $idx. $var1_display vs $var2_display $type_str: r=$(round(corr_val, digits=4))$sig_str")
end

println("\n✅ Cross-correlation analysis complete!")
println("   - LaTeX table: $latex_path")
println("   - Input-output heatmap: $heatmap_path")
println("   - Output-output heatmap: $output_heatmap_path")
println("   - Combined heatmap: $combined_heatmap_path")
if !isempty(all_scatter_plots)
    println("   - Scatter plots: $(joinpath(output_dir, "cross_correlations_scatter_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png"))")
end

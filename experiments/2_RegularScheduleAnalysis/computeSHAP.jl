# SHAP Analysis for Model Input-Output Relationships
# Lara Weed
# Generate SHAP beehive plot to explain model predictions.
#
# Paths from jetlag_reference.jl: particle summaries in Data/JetLag/2_RegularScheduleAnalysis/particleSummaries/,
# outputs to regular_schedule_shap_dir().
#
# This script:
# 1. Loads particle summaries (same as computeCrossCorrelations.jl)
# 2. Trains a model to predict outputs from inputs
# 3. Computes SHAP values to explain predictions
# 4. Creates a beehive plot showing feature importance
# 5. Writes one CSV per output variable: mean |SHAP| by feature (sorted), see regular_schedule_shap_dir(condition)
# 6. Writes wide mean |SHAP| CSV matrix (inputs × outputs); combined heatmap figure (Fig 2): plotCorrelationsAndSHAPCombined.jl

using Pkg
using CircadianPOMDP

using JLD2
using Plots
using Statistics
using StatsBase
using DataFrames
using Printf
using Dates
using Random

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

# Try to load PyCall for SHAP analysis
try
    using PyCall
    global HAS_PYCALL = true
    println("✓ PyCall loaded successfully")
catch
    @warn "PyCall not available. Attempting to install..."
    try
        Pkg.add("PyCall")
        using PyCall
        global HAS_PYCALL = true
        println("✓ PyCall installed and loaded")
    catch e
        error("Could not install/load PyCall: $e\nPlease install manually: using Pkg; Pkg.add(\"PyCall\")")
    end
end

# Install Python packages if needed
if HAS_PYCALL
    try
        # Try to import SHAP - check if already installed
        # First, ensure user site-packages are prioritized
        py"""
        import sys
        import os
        
        # Add user site-packages to the front of sys.path if not already there
        import site
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
        
        try:
            import shap
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            print("All required Python packages are already installed.")
        except (ImportError, ValueError) as e:
            # ValueError can occur with binary incompatibilities
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
            print(f"Import error: {missing_pkg}")
            raise
        """
        global HAS_SHAP = true
        println("✓ Python SHAP library available")
    catch e
        @warn "Could not import SHAP. Attempting to install Python packages..."
        println("   Note: This may require sudo privileges or a virtual environment.")
        println("   If installation fails, please install manually:")
        println("   pip install --user shap scikit-learn pandas numpy")
        println("   or: python3 -m pip install --user shap scikit-learn pandas numpy")
        
        try
            # Try with --user and --break-system-packages flags (for externally-managed environments)
            py"""
            import subprocess
            import sys
            import os
            
            # Try --user install with --break-system-packages first (for externally-managed environments)
            try:
                print("Attempting to install with --user --break-system-packages flags...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "shap", "scikit-learn", "pandas", "numpy"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    print(f"Installation output: {result.stdout}")
                    print(f"Installation errors: {result.stderr}")
                    # Check if it's the externally-managed-environment error
                    if "externally-managed-environment" in result.stderr:
                        raise RuntimeError("Externally-managed environment detected. Please install manually.")
                    raise RuntimeError(f"pip install failed with return code {result.returncode}")
                print("Installation successful!")
            except Exception as e:
                print(f"User install failed: {e}")
                # Try without --user flag (may require sudo)
                print("Attempting system-wide install (may require sudo)...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "shap", "scikit-learn", "pandas", "numpy"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    print(f"Installation output: {result.stdout}")
                    print(f"Installation errors: {result.stderr}")
                    raise RuntimeError(f"pip install failed with return code {result.returncode}")
                print("Installation successful!")
            """
            
            # Now try importing again
            py"""
            import shap
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            """
            global HAS_SHAP = true
            println("✓ Python SHAP library installed and loaded")
        catch e2
            error("""
Could not install Python SHAP library automatically.

Please install manually using one of these methods:

1. User installation (recommended, for externally-managed environments):
   python3 -m pip install --user --break-system-packages shap scikit-learn pandas numpy

2. System-wide installation (requires sudo):
   sudo python3 -m pip install --break-system-packages shap scikit-learn pandas numpy

3. Using conda (if you use conda):
   conda install -c conda-forge shap scikit-learn pandas numpy

After installation, run this script again.

Error details: $e2
""")
        end
    end
end

# ================================================================
# MAIN SCRIPT
# ================================================================

println("🔍 SHAP Analysis for Model Input-Output Relationships")
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

# --- Prepare Data for SHAP Analysis ---
println("\n3. Preparing data for SHAP analysis...")

# Define input and output names
input_names = ["tau", "k", "psi", "chi", "mu_h_over_chi"]
input_labels_plot = ["τ", "k", "ψ", "χ", "μ_H/χ"]
input_data = [tau_values, k_values, psi_values, chi_values, mu_h_over_chi_values]
keep = varying_inputs(input_data)   # ψ drops out when fixed at 0
input_names, input_labels_plot, input_data = input_names[keep], input_labels_plot[keep], input_data[keep]

output_names = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "psi_obs"]
output_labels_plot = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"]
output_data = [s_on_values, s_off_values, s_dur_values, s_mp_values, cbtmin_values, psi_obs_values]

# Create feature matrix (inputs) and target vectors (outputs)
n_samples = length(tau_values)
n_features = length(input_names)
n_outputs = length(output_names)

# Build feature matrix
X = zeros(n_samples, n_features)
for i in 1:n_features
    X[:, i] = input_data[i]
end

# Build target matrix
Y = zeros(n_samples, n_outputs)
for i in 1:n_outputs
    Y[:, i] = output_data[i]
end

# Filter out any rows with NaN or Inf values
valid_mask = trues(n_samples)
for i in 1:n_samples
    for j in 1:n_features
        if !isfinite(X[i, j])
            valid_mask[i] = false
            break
        end
    end
    if valid_mask[i]
        for j in 1:n_outputs
            if !isfinite(Y[i, j])
                valid_mask[i] = false
                break
            end
        end
    end
end

X_clean = X[valid_mask, :]
Y_clean = Y[valid_mask, :]
n_clean = size(X_clean, 1)

println("   ✓ Prepared data: $n_clean samples, $n_features features, $n_outputs outputs")
println("   ✓ Removed $(n_samples - n_clean) samples with invalid values")

# --- Train Models and Compute SHAP Values ---
println("\n4. Training models and computing SHAP values...")

# Convert to Python arrays using PyCall
# We'll pass data directly in the py""" blocks below

# Compute SHAP values for each output
shap_values_all = Dict()
shap_explainers = Dict()

for output_idx in 1:n_outputs
    output_name = output_names[output_idx]
    output_label = output_labels_plot[output_idx]
    
    println("   Computing SHAP for $output_label...")
    
    try
        # Convert data to Python arrays for this iteration
        X_py_obj = PyObject(X_clean)
        Y_py_obj = PyObject(Y_clean)
        output_idx_py = output_idx - 1  # Python is 0-indexed
        
        # Train a Random Forest model and compute SHAP values
        py"""
        import numpy as np
        import shap
        from sklearn.ensemble import RandomForestRegressor
        
        # Get the target for this output
        X_data = $X_py_obj
        Y_data = $Y_py_obj
        output_idx = $output_idx_py
        
        y_target = Y_data[:, output_idx]
        
        # Train Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_data, y_target)
        
        # Create SHAP explainer (using TreeExplainer for Random Forest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        
        # Store in Python namespace for retrieval (use a fixed name)
        import __main__
        __main__.shap_values_result = shap_values
        """
        
        # Get SHAP values back to Julia
        shap_vals_py = py"__main__.shap_values_result"
        shap_vals_julia = Array{Float64}(shap_vals_py)
        shap_values_all[output_name] = shap_vals_julia
        
        println("      ✓ Computed SHAP values for $output_label")
        
    catch e
        @warn "Error computing SHAP for $output_label: $e"
    end
end

println("   ✓ Computed SHAP values for all outputs")

# --- Create Beehive Plot ---
println("\n5. Creating SHAP beehive plot...")

# Publication-ready settings
default(fontfamily="Times", 
        linewidth=2.5,
        markersize=6,
        dpi=300)

# Beeswarm figure (S2/S3) in the manuscript style (figure_style.jl); aspect ratio of the manuscript figure (730 × 611)
const BEE_FIG_W = 1600
const BEE_TICK_FS, BEE_GUIDE_FS, BEE_PANEL_FS = 20, 22, 24   # printed at 6.5 in wide (SI): 8.1, 8.9, 9.8 pt
"""SHAP value range (min, max over all outputs) of a condition's last run, shared by the S4/S5 x axes."""
shap_range_path(c) = joinpath(regular_schedule_shap_dir(; condition=c), "shap_value_range_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).csv")

# Create a combined beehive plot showing all outputs
function create_shap_beehive_plot(shap_values_dict, feature_names, output_names, output_labels, X_data)
    # Calculate mean absolute SHAP values for each feature across all outputs
    feature_importance = Dict()
    for feat in feature_names
        feature_importance[feat] = 0.0
    end
    
    for (output_name, shap_vals) in shap_values_dict
        if isa(shap_vals, Array)
            mean_abs_shap = mean(abs.(shap_vals), dims=1)
            for (idx, feat) in enumerate(feature_names)
                feature_importance[feat] += mean_abs_shap[idx]
            end
        end
    end
    
    # Sort features by importance
    sorted_features = sort(collect(feature_names), by=x -> feature_importance[x], rev=true)
    
    # Determine global color scale limits (0 to 1 since we normalize)
    # This ensures consistent coloring across all plots
    global_clims = (0.0, 1.0)
    
    # Calculate global x-axis limits from all SHAP values
    # This ensures all plots use the same x-axis scale
    # Use actual min/max from data (not symmetric)
    all_shap_values = Float64[]
    for (output_name, shap_vals) in shap_values_dict
        if isa(shap_vals, Array)
            append!(all_shap_values, vec(shap_vals))
        end
    end
    
    # Use actual min/max with some padding
    if !isempty(all_shap_values)
        min_shap, max_shap = extrema(all_shap_values)
        # Same x axis in both light conditions (S4 and S5): save this condition's range and use the union with the
        # other condition's saved range. Run both conditions, then rerun the first so it sees the second's range.
        mkpath(dirname(shap_range_path(condition)))
        open(io -> println(io, "min,max\n$min_shap,$max_shap"), shap_range_path(condition), "w")
        for c in REGULAR_SCHEDULE_CONDITIONS
            c == condition && continue
            if isfile(shap_range_path(c))
                lo, hi = parse.(Float64, split(readlines(shap_range_path(c))[2], ","))
                min_shap, max_shap = min(min_shap, lo), max(max_shap, hi)
            else
                @warn "No saved SHAP range for $c: x axis not shared with it (run $c, then rerun $condition)"
            end
        end
        # Add 5% padding on each side
        range_shap = max_shap - min_shap
        padding = range_shap * 0.05
        global_xlims = (min_shap - padding, max_shap + padding)
    else
        global_xlims = (-1.0, 1.0)  # Default fallback
    end
    
    # Create subplots for each output
    plots_list = []
    
    for (output_idx, output_name) in enumerate(output_names)
        if haskey(shap_values_dict, output_name)
            shap_vals = shap_values_dict[output_name]
            output_label = output_labels[output_idx]
            
            # Convert to matrix if needed
            if isa(shap_vals, Vector)
                shap_vals = reshape(shap_vals, length(shap_vals), 1)
            end
            
            # Get feature values for coloring
            feature_values = X_data
            
            # Create data for plotting with jitter for better visualization
            n_samples = size(shap_vals, 1)
            n_feat = size(shap_vals, 2)
            
            # For each feature, create a scatter plot with jitter
            y_positions = Float64[]
            shap_plot_vals = Float64[]
            feature_plot_vals = Float64[]
            
            rng = MersenneTwister(42)  # For reproducible jitter
            
            for (feat_idx, feat_name) in enumerate(sorted_features)
                # Find original index
                orig_idx = findfirst(x -> x == feat_name, feature_names)
                if orig_idx !== nothing && orig_idx <= n_feat
                    # Get SHAP values for this feature
                    shap_feat = shap_vals[:, orig_idx]
                    feat_vals = feature_values[:, orig_idx]
                    
                    # Normalize feature values for better color mapping (percentile-based)
                    p5 = percentile(feat_vals, 5)
                    p95 = percentile(feat_vals, 95)
                    if p95 > p5
                        normalized_feat = (feat_vals .- p5) ./ (p95 - p5)
                        normalized_feat = clamp.(normalized_feat, 0.0, 1.0)
                    else
                        normalized_feat = fill(0.5, length(feat_vals))
                    end
                    
                    # Add jitter to y-positions for better visualization (beehive effect)
                    jitter_amount = 0.3
                    y_jitter = feat_idx .+ (rand(rng, n_samples) .- 0.5) .* jitter_amount
                    
                    # Append to plot data
                    append!(y_positions, y_jitter)
                    append!(shap_plot_vals, shap_feat)
                    append!(feature_plot_vals, normalized_feat)
                end
            end
            
            # Output name inside the panel (top left); feature labels on the left column and the x axis on the
            # bottom row only (3 × 2 panels)
            left_col, bottom_row = isodd(output_idx), output_idx > length(output_names) - 2
            n_sf = length(sorted_features)
            p = scatter(shap_plot_vals, y_positions,
                       marker_z=feature_plot_vals, color=:Greys, clims=global_clims, alpha=0.8,
                       markersize=2.0, markerstrokewidth=0,
                       xlabel=(bottom_row ? "SHAP Value" : ""), ylabel="", title="",
                       yticks=(1:n_sf, left_col ? display_label.(sorted_features) : fill("", n_sf)),
                       xformatter=(bottom_row ? :auto : (_ -> "")),
                       yflip=true, colorbar=false, legend=false,
                       grid=true, gridwidth=1.0, gridalpha=0.4, framestyle=:box,
                       guidefont=(FIG_BOLD, BEE_GUIDE_FS), tickfont=(FIG_BOLD, BEE_TICK_FS),
                       left_margin=(left_col ? 12Plots.mm : 1Plots.mm), right_margin=2Plots.mm,
                       top_margin=1Plots.mm, bottom_margin=(bottom_row ? 9Plots.mm : 1Plots.mm),
                       xlims=global_xlims, ylims=(0.15, n_sf + 0.5))

            # Add vertical line at x=0
            vline!(p, [0], linestyle=:dash, linecolor=:black, linewidth=1.5, alpha=0.7, label="")
            annotate!(p, global_xlims[1] + 0.02 * (global_xlims[2] - global_xlims[1]), 0.50,
                      text(display_label(output_label), FIG_BOLD, BEE_PANEL_FS, :left))

            push!(plots_list, p)
        end
    end
    isempty(plots_list) && error("No plots created")

    # Grey scale bar for the (normalised) feature value, top right
    zbar = collect(range(global_clims[1], global_clims[2], length=101))
    cb = heatmap([0.0], zbar, reshape(zbar, :, 1); color=:Greys, clims=global_clims, colorbar=false,
                 xticks=false, yticks=false, framestyle=:box, legend=false, xlims=(-0.5, 0.5), ylims=(0, 1),
                 left_margin=24Plots.mm, right_margin=16Plots.mm, top_margin=34Plots.mm, bottom_margin=6Plots.mm)
    annotate!(cb, -0.9, 1.0, text("1", FIG_BOLD, BEE_TICK_FS, :right))
    annotate!(cb, -0.9, 0.0, text("0", FIG_BOLD, BEE_TICK_FS, :right))
    annotate!(cb, 0.0, 1.24, text("Feature", FIG_BOLD, BEE_GUIDE_FS))
    annotate!(cb, 0.0, 1.10, text("Value", FIG_BOLD, BEE_GUIDE_FS))
    blank = plot(framestyle=:none, showaxis=false, grid=false, legend=false)
    panels = [plots_list[1], plots_list[2], cb, plots_list[3], plots_list[4], blank, plots_list[5], plots_list[6], blank]
    return plot(panels...; layout=grid(3, 3; widths=[0.445, 0.445, 0.11]),
                size=(BEE_FIG_W, round(Int, BEE_FIG_W * 611 / 730)))
end

# Create the beehive plot
beehive_plot = create_shap_beehive_plot(shap_values_all, input_labels_plot, output_names, output_labels_plot, X_clean)

# Save the plot
output_dir = regular_schedule_shap_dir(; condition)
mkpath(output_dir)
beehive_path = joinpath(output_dir, "shap_beehive_plot_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.png")
savefig(beehive_plot, beehive_path)
println("   ✓ Saved SHAP beehive plot to: $beehive_path")

# --- Mean |SHAP| heatmap (inputs × outputs), same layout style as cross-correlation heatmap ---
println("\n5b. Mean |SHAP| input–output matrix (CSV; combined heatmap: plotCorrelationsAndSHAPCombined.jl)...")

"""Build matrix [input, output] of mean absolute SHAP; NaN if output missing."""
function mean_abs_shap_matrix(shap_dict, output_names_order, n_feat::Int)
    n_out = length(output_names_order)
    M = fill(NaN, n_feat, n_out)
    for (j, output_name) in enumerate(output_names_order)
        sh = get(shap_dict, output_name, nothing)
        if isa(sh, Array) && size(sh, 2) == n_feat
            M[:, j] = vec(mean(abs.(sh), dims=1))
        end
    end
    return M
end

shap_heatmap_mat = mean_abs_shap_matrix(shap_values_all, output_names, n_features)
combined_csv = joinpath(output_dir, "shap_mean_abs_matrix_$(condition)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).csv")
open(combined_csv, "w") do io
    print(io, "input_feature")
    for oname in output_names
        print(io, ",", oname)
    end
    println(io)
    for (ii, feat) in enumerate(input_labels_plot)
        print(io, feat)
        for j in 1:length(output_names)
            v = shap_heatmap_mat[ii, j]
            print(io, ",", isfinite(v) ? v : "")
        end
        println(io)
    end
end
println("   ✓ Saved combined matrix CSV: $(basename(combined_csv))")

# --- Print summary statistics and save mean |SHAP| per feature (one CSV per output variable) ---
# CSVs live under regular_schedule_shap_dir(condition), so the light diet is encoded in the path
# (e.g. .../LowLight150lux/SHAP/). Filenames: shap_mean_abs_<output_key>_<alignment_suffix>.csv
println("\n7. SHAP summary statistics (console + CSV per output)")
println("="^70)

for (output_name, shap_vals) in shap_values_all
    output_idx = findfirst(x -> x == output_name, output_names)
    output_label = output_labels_plot[output_idx]
    
    println("\n$output_label:")
    
    if isa(shap_vals, Array)
        mean_abs_shap = vec(mean(abs.(shap_vals), dims=1))
        feature_importance = [(input_labels_plot[i], mean_abs_shap[i]) for i in eachindex(input_labels_plot)]
        sort!(feature_importance, by=x -> x[2], rev=true)
        
        for (feat_name, importance) in feature_importance
            println("  $feat_name: $(round(Float64(importance), digits=4))")
        end

        csv_path = joinpath(
            output_dir,
            "shap_mean_abs_$(output_name)_$(condition)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).csv",
        )
        open(csv_path, "w") do io
            println(io, "output_key,output_label,light_diet_condition,feature,mean_abs_shap")
            for (feat_name, importance) in feature_importance
                imp = Float64(importance)
                println(io, output_name, ",", output_label, ",", condition, ",", feat_name, ",", imp)
            end
        end
        println("  → CSV: $(basename(csv_path))")
    end
end

println("\n✅ SHAP analysis complete!")
println("   - Beehive plot: $beehive_path")
println("   - Mean |SHAP| matrix CSV: $combined_csv")
println("   - Combined heatmaps (both light diets, shared color scale): run plotCorrelationsAndSHAPCombined.jl (Fig 2)")
println("   - Mean |SHAP| tables (per output): $(output_dir)/shap_mean_abs_*_$(condition)_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX).csv")

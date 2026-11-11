# Figures for reviewer comment 3.6 (R3.7 in the response letter) from rf_fit_and_shap_stability.jl's results:
#   figures/rf_cv_r2.png       : R² per output — RF out-of-fold (mean ± SD, 50 fits), linear out-of-fold, RF in-sample
#   figures/shap_stability.png : mean |SHAP| (h) per input — 10 seeds and 50 held-out folds (mean, min–max bars),
#                                interventional SHAP (seed 42)
# Drawn at print size (figure_style.jl): at the SI width: R² half page (6.5 × 3.4 in), SHAP stability full page (6.5 × 7 in); text 8–9 pt.
# Run: julia --project=. experiments/2b_SensitivityRevision/plot_rf_fit_and_shap_stability.jl
#      (also run at the end of rf_fit_and_shap_stability.jl)

using CSV
using DataFrames
using Plots
include(joinpath(@__DIR__, "..", "figure_style.jl"))

const PLOT_RESULTS_DIR = joinpath(@__DIR__, "results")
const PLOT_FIG_DIR = joinpath(@__DIR__, "figures")
mkpath(PLOT_FIG_DIR)
const PLOT_CONDS = ("LowLight150lux", "HighLight10000lux")
const COND_TITLE = Dict("LowLight150lux" => "150 lux", "HighLight10000lux" => "10,000 lux")
const PLOT_INPUTS = unique(CSV.read(joinpath(PLOT_RESULTS_DIR, "shap_stability.csv"), DataFrame).input)   # ψ absent when fixed at 0
const PLOT_OUTPUTS = ["S_On", "S_Off", "S_Dur", "S_MP", "CBT_min", "ψ_obs"]
# Categorical palette, fixed order (as before); markers also differ
const COL_OOF, COL_LIN, COL_IN = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"

default(fontfamily=FIG_BOLD, framestyle=:box, dpi=300, grid=false, tickfont=(FIG_BOLD, PT_TICK),
        guidefont=(FIG_BOLD, PT_GUIDE), titlefont=(FIG_BOLD, PT_LABEL), legendfont=(FIG_BOLD, PT_LEGEND),
        background_color=:white)

"""A legend-only panel (no axes) for a row under the figure."""
function legend_row(entries; columns)
    p = plot(framestyle=:none, showaxis=false, grid=false, legend=:top, legend_column=columns,
             xlims=(0, 1), ylims=(0, 1), foreground_color_legend=nothing)
    for (lab, col, mk) in entries
        scatter!(p, [-1.0], [0.0]; color=col, markerstrokecolor=col, markershape=mk, markersize=3.5, label=lab)
    end
    return p
end

fit_df = CSV.read(joinpath(PLOT_RESULTS_DIR, "rf_cv_r2.csv"), DataFrame)
shap_df = CSV.read(joinpath(PLOT_RESULTS_DIR, "shap_stability.csv"), DataFrame)

# --- Figure 1: R² per output, one panel per light condition ---
r2_panels = map(enumerate(PLOT_CONDS)) do (ci, c)
    d = fit_df[fit_df.condition .== c, :]
    y = collect(1:nrow(d))
    p = scatter(d.rf_oof_mean, y .+ 0.15; xerror=d.rf_oof_sd, color=COL_OOF, markerstrokecolor=COL_OOF, markersize=3.5,
                yticks=(y, ci == 1 ? display_label.(d.output) : fill("", nrow(d))), xlims=(0.7, 1.005), xticks=0.7:0.05:1.0,   # all values lie in 0.73–1.00
                ylims=(0.4, nrow(d) + 0.6), legend=false, title=COND_TITLE[c], titlelocation=:left, xlabel="R²",
                left_margin=(ci == 1 ? 2Plots.mm : 0Plots.mm), bottom_margin=2Plots.mm)
    scatter!(p, d.linear_oof_mean, y; color=COL_LIN, markerstrokecolor=COL_LIN, markershape=:diamond, markersize=3.5)
    scatter!(p, d.rf_insample, y .- 0.15; color=COL_IN, markerstrokecolor=COL_IN, markershape=:utriangle, markersize=3.5)
    p
end
# One column: GR legend columns all take the longest entry's width, so a multi-column legend of unequal entries is not
# centred under the panels
leg1 = legend_row([("RF, out-of-fold (mean ± SD, 50 fits)", COL_OOF, :circle),
                   ("Linear regression, out-of-fold", COL_LIN, :diamond),
                   ("RF, in-sample (model in the paper)", COL_IN, :utriangle)]; columns=1)
savefig(plot(plot(r2_panels...; layout=(1, 2)), leg1; layout=@layout([a{0.74h}; b{0.26h}]),
             size=print_size(SI_WIDTH_IN, 3.4)), joinpath(PLOT_FIG_DIR, "rf_cv_r2.png"))

# --- Figure 2: mean |SHAP| per input; 4 × 3 panels (rows 1–2 = 150 lux, rows 3–4 = 10,000 lux) ---
shap_panels = []
for (ci, c) in enumerate(PLOT_CONDS), (oi, o) in enumerate(PLOT_OUTPUTS)
    d = shap_df[(shap_df.condition .== c) .& (shap_df.output .== o), :]
    s, fo, iv = (d[d.source .== src, :] for src in ("seeds", "folds", "interventional"))
    x = collect(1:length(PLOT_INPUTS))
    left, bottom = mod1(oi, 3) == 1, ci == 2 && oi > 3
    p = scatter(x .- 0.18, s.mean; yerror=(s.mean .- s.min, s.max .- s.mean), color=COL_OOF, markerstrokecolor=COL_OOF,
                markersize=2.5, xticks=(x, bottom ? display_label.(PLOT_INPUTS) : fill("", length(x))),
                xlims=(0.4, length(x) + 0.6), ylims=(0, 1.15 * maximum(vcat(s.max, fo.max, iv.mean))), legend=false,
                title="$(COND_TITLE[c]), $(display_label(o))", titlelocation=:left,
                ylabel=(left ? "Mean |SHAP| (h)" : ""), left_margin=(left ? 2Plots.mm : 0Plots.mm),
                bottom_margin=(bottom ? 1Plots.mm : 0Plots.mm))
    scatter!(p, x, fo.mean; yerror=(fo.mean .- fo.min, fo.max .- fo.mean), color=COL_LIN, markerstrokecolor=COL_LIN,
             markershape=:diamond, markersize=2.5)
    scatter!(p, x .+ 0.18, iv.mean; color=COL_IN, markerstrokecolor=COL_IN, markershape=:utriangle, markersize=2.5)
    push!(shap_panels, p)
end
leg2 = legend_row([("10 seeds, in-sample (mean, min–max)", COL_OOF, :circle),
                   ("50 held-out folds (mean, min–max)", COL_LIN, :diamond),
                   ("Interventional SHAP (seed 42)", COL_IN, :utriangle)]; columns=1)
savefig(plot(plot(shap_panels...; layout=(4, 3)), leg2; layout=@layout([a{0.91h}; b{0.09h}]),
             size=print_size(SI_WIDTH_IN, FULL_PAGE_H_IN)), joinpath(PLOT_FIG_DIR, "shap_stability.png"))
println("Wrote figures/rf_cv_r2.png and figures/shap_stability.png")

# Reviewer comment 2.4 figures from the SD sweep (sd_sweep.jl → results/sd_sweep_all.csv).
#   figures/sd_sweep_influence.png : |Spearman ρ| (95 % bootstrap CI) of each parameter with each output as that
#                                    parameter's own prior SD is scaled (×0.5–2), per light condition; ψ (fixed at 0 in
#                                    the baseline) at an added SD of 0.125–2 h, plotted at x = SD in hours.
#   figures/sd_sweep_top_input_{spearman,shap}.png : top input by |ρ| / by mean |SHAP| for every scenario × output ×
#                                    condition, named in the baseline row and where it differs from baseline (shaded).
# Paper style (figure_style.jl), drawn at print size: full page at the SI width (6.5 × 6.55 in), text 8–9 pt; parameter colours match Fig 4.
# Run: julia --project=. experiments/2b_SensitivityRevision/plot_sd_sweep.jl

using CSV
using DataFrames
using Plots
using Printf
include(joinpath(@__DIR__, "..", "figure_style.jl"))

const B = @__DIR__
const FIG_DIR = joinpath(B, "figures")
mkpath(FIG_DIR)
d = CSV.read(joinpath(B, "results", "sd_sweep_all.csv"), DataFrame)

const CONDS = ["LowLight150lux", "HighLight10000lux"]
const COND_TITLE = Dict("LowLight150lux" => "150 lux", "HighLight10000lux" => "10,000 lux")
const OUTS = ["s_on", "s_off", "s_dur", "s_mp", "cbtmin", "psi_obs"]
const OUT_LABEL = Dict(o => display_label(l) for (o, l) in ("s_on" => "S_On", "s_off" => "S_Off", "s_dur" => "S_Dur",
                       "s_mp" => "S_MP", "cbtmin" => "CBT_min", "psi_obs" => "ψ_obs"))
const IN_LABEL = Dict(i => display_label(l) for (i, l) in ("tau" => "τ", "k" => "k", "psi" => "ψ", "chi" => "χ",
                      "mu_h_over_chi" => "μ_H/χ"))
# Varied prior → model input whose influence is shown (μ_H enters the model as μ_H/χ, φ as ψ); colours as Fig 4
const VARIED = [("tau_x", "tau", raw"\tau", "#1565A8", :circle), ("K", "k", raw"\kappa", "#00796B", :square),
                ("phi", "psi", raw"ψ (added SD, h)", "#6A1B9A", :diamond), ("chi", "chi", raw"\chi", "#AD1457", :utriangle),
                ("mu_h", "mu_h_over_chi", "\\mu_{H}\u2215\\chi (\\mu_{H} prior)", "#C44F17", :dtriangle)]

default(fontfamily=FIG_BOLD, framestyle=:box, dpi=300, grid=true, gridalpha=0.2, tickfont=(FIG_BOLD, PT_TICK),
        guidefont=(FIG_BOLD, PT_GUIDE), titlefont=(FIG_BOLD, PT_LABEL), legendfont=(FIG_BOLD, PT_LEGEND),
        background_color=:white)

"""Scale of the varied prior in a scenario (1 = baseline); φ as the added ψ SD in hours."""
function scale_of(row, varied)
    varied == "phi" ? row.phi_sd_h :
    varied == "tau_x" ? row.scale_tau : varied == "K" ? row.scale_K : varied == "chi" ? row.scale_chi : row.scale_mu_h
end

# ---------------- Figure A: influence vs prior width ----------------
const S2_H_IN = 6.55   # full page (≤ FULL_PAGE_H_IN) with the legend close under the x labels
panels = []
for (ci, c) in enumerate(CONDS), (oi, o) in enumerate(OUTS)
    bottom = ci == 2 && oi > 3   # 4 × 3 grid: rows 1–2 = 150 lux, rows 3–4 = 10,000 lux
    p = plot(xscale=:log2, xticks=([0.125, 0.25, 0.5, 1, 2], ["1/8", "1/4", "1/2", "1", "2"]), xlims=(0.11, 2.3), ylims=(0, 1.05),
             title="$(COND_TITLE[c]), $(OUT_LABEL[o])", titlelocation=:left,
             xlabel=(bottom ? "SD Scale (× Baseline); ψ: SD (h)" : ""), xformatter=(bottom ? :auto : (_ -> "")),
             ylabel=(mod1(oi, 3) == 1 ? "|ρ|" : ""), legend=false,
             yformatter=(mod1(oi, 3) == 1 ? :auto : (_ -> "")),   # shared 0–1 scale: tick labels on the first column only
             left_margin=(mod1(oi, 3) == 1 ? 2Plots.mm : 0Plots.mm), right_margin=0Plots.mm, bottom_margin=0Plots.mm)
    vline!(p, [1.0]; color=:gray60, linestyle=:dash, linewidth=1, label="")
    for (varied, input, lbl, col, mk) in VARIED
        g = d[(d.condition .== c) .& (d.output .== o) .& (d.input .== input) .&
              ((d.varied .== varied) .| (d.scenario .== "baseline")), :]   # ψ has no baseline row (fixed at 0)
        g = g[g.distribution .== "normal", :]
        x = [scale_of(r, varied) for r in eachrow(g)]
        ord = sortperm(x)
        y, lo, hi = abs.(g.rho[ord]), g.rho_lo[ord], g.rho_hi[ord]
        elo = y .- min.(abs.(lo), abs.(hi)); ehi = max.(abs.(lo), abs.(hi)) .- y
        # (CIs straddling 0 are drawn from 0)
        elo .= [ (l <= 0 <= h) ? yy : e for (l, h, yy, e) in zip(lo, hi, y, elo)]
        plot!(p, x[ord], y; yerror=(elo, ehi), color=col, marker=mk, markersize=2.5, markerstrokecolor=col,
              linewidth=1.0, label=(ci == 1 && oi == 1 ? lbl : ""))
    end
    push!(panels, p)
end
legend_panel = plot(framestyle=:none, grid=false, showaxis=false, legend=:top, legend_column=3,
                    xlims=(0, 1), ylims=(0, 1), foreground_color_legend=nothing,
                    # GR places a :top legend well below the strip top; pull it up to ~0.17 in under the x labels
                    # (transparent, so the strip cannot cover the labels)
                    top_margin=-3Plots.mm, background_color_subplot=:transparent)
# GR legend columns all take the longest entry's width: put the longest ("µH/χ (µH prior)") at the end of the first row
# so the legend is centred
for k in (1, 2, 5, 4, 3)
    (_, _, lbl, col, mk) = VARIED[k]
    plot!(legend_panel, [-1.0], [0.0]; color=col, marker=mk, markersize=3, markerstrokecolor=col, linewidth=1.0, label=lbl)
end
fig = plot(plot(panels...; layout=(4, 3)), legend_panel; layout=@layout([a{0.965h}; b{0.035h}]),
           size=print_size(SI_WIDTH_IN, S2_H_IN))
savefig(fig, joinpath(FIG_DIR, "sd_sweep_influence.png"))

# ---------------- Figure B: top input per scenario ----------------
const SCEN_ORDER = vcat(["baseline"], ["tau_x_x$(s)" for s in (0.5, 1.5, 2.0)], ["K_x$(s)" for s in (0.5, 1.5, 2.0)],
                        ["chi_x$(s)" for s in (0.5, 1.5, 2.0)], ["mu_h_x$(s)" for s in (0.5, 1.5, 2.0)],
                        ["phi_sd$(s)h" for s in (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)], ["all_x0.5", "all_x2.0", "uniform"])
scen_label(s) = s == "baseline" ? "baseline" : s == "uniform" ? "uniform prior" :
    startswith(s, "phi_sd") ? raw"\psi SD " * replace(s, "phi_sd" => "", "h" => " h") :
    replace(s, "tau_x_x" => raw"\tau × ", "K_x" => raw"\kappa × ", "chi_x" => raw"\chi × ",
            "mu_h_x" => raw"\mu_{H} × ", "all_x" => "all × ")
top_input(g, col) = IN_LABEL[g.input[argmax(abs.(g[!, col]))]]

# One figure per metric (Spearman |ρ|, mean |SHAP|), each 150 lux | 10,000 lux, sized for the supplement's text width
# (6.0 × 4.2 in, text 9–10 pt). The baseline row names the top input; other rows show it only where it differs from
# baseline (shaded), so a blank cell means "same as baseline".
const TOP_W_IN, TOP_H_IN = SI_WIDTH_IN, 4.2
const TOP_FS, TOP_TITLE_FS = 9, 10
for (metric, col, mlabel, file) in (("rho", :rho, "Spearman |ρ|", "sd_sweep_top_input_spearman.png"),
                                    ("shap", :mean_abs_shap, "Mean |SHAP|", "sd_sweep_top_input_shap.png"))
    blocks = []
    for c in CONDS
        M = fill("", length(SCEN_ORDER), length(OUTS)); diff = zeros(length(SCEN_ORDER), length(OUTS))
        for (r, s) in enumerate(SCEN_ORDER), (k, o) in enumerate(OUTS)
            g = d[(d.scenario .== s) .& (d.condition .== c) .& (d.output .== o), :]
            gb = d[(d.scenario .== "baseline") .& (d.condition .== c) .& (d.output .== o), :]
            M[r, k] = top_input(g, col)
            diff[r, k] = M[r, k] == top_input(gb, col) ? 0.0 : 1.0
        end
        p = heatmap(1:length(OUTS), 1:length(SCEN_ORDER), diff; color=cgrad(["#FFFFFF", "#F2C2A0"]), clims=(0, 1),
                    colorbar=false, yflip=true, xticks=(1:length(OUTS), [OUT_LABEL[o] for o in OUTS]),
                    yticks=(1:length(SCEN_ORDER), c == CONDS[1] ? scen_label.(SCEN_ORDER) : fill("", length(SCEN_ORDER))),
                    title="$mlabel, $(COND_TITLE[c])", titlelocation=:left, grid=false,
                    titlefont=(FIG_BOLD, TOP_TITLE_FS), ytickfont=(FIG_BOLD, TOP_FS), xtickfont=(FIG_BOLD, TOP_FS - 1),
                    left_margin=(c == CONDS[1] ? 2Plots.mm : 0Plots.mm), bottom_margin=1Plots.mm)
        hline!(p, [1.5, 4.5, 7.5, 10.5, 13.5, 20.5]; color=:gray55, linewidth=0.6, label="")
        for r in eachindex(SCEN_ORDER), k in eachindex(OUTS)
            (r == 1 || diff[r, k] > 0) && annotate!(p, k, r, text(M[r, k], FIG_BOLD, TOP_FS, :black))
        end
        push!(blocks, p)
    end
    savefig(plot(blocks...; layout=(1, 2), size=print_size(TOP_W_IN, TOP_H_IN)), joinpath(FIG_DIR, file))
end
println("Wrote figures/sd_sweep_influence.png, sd_sweep_top_input_spearman.png and sd_sweep_top_input_shap.png")

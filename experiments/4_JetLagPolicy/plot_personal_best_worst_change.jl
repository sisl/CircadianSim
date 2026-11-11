# Supplementary figure (S13): per-participant effect of the oracle personal best and worst light windows, by shift.
#   Top row:    change in adaptation time vs no intervention (days; negative = faster): median of each participant's
#               change, shaded over the central 95 % of participants (2.5th–97.5th percentiles); baseline at 0.
#   Middle row: change as a percentage of each participant's own baseline adaptation time (100 × change / baseline),
#               median and central 95 % of participants; baseline at 0 %.
#   Bottom row: number of participants slower with their personal worst (above 0) and faster with their personal best
#               (below 0, plotted as negative so the two do not overlap; tick labels show counts), mirroring the top row.
# Shading shows how the change varies across the population. Alternatives (separate files): "iqr" shades the
# interquartile range, "ci" the 95 % bootstrap confidence interval of the median (participants resampled N_BOOT times;
# best and worst from the same resample), which is very narrow at n = 967. All three are written to the CSV.
#   Columns:    150 lux | 10,000 lux.
# Oracle personal best = each participant's best option, any light window or no intervention; oracle personal worst =
# each participant's worst light window (as in Fig 5). Non-adapters count at the post-shift window length.
# Drawn at the SI width (6.5 × 6.85 in, figure_style.jl), text 8–10 pt; line key below the panels. Also writes the numbers per shift to a CSV.
# Run: julia --project=. experiments/4_JetLagPolicy/plot_personal_best_worst_change.jl

include(joinpath(@__DIR__, "jetlag_load_steady_state.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))
using Plots, Printf, DataFrames, CSV, Random

const SHADE = "iqr" in ARGS ? :iqr : "ci" in ARGS ? :ci : :p95
const BAND = Dict(:p95 => ("p025", "p975"), :iqr => ("q25", "q75"), :ci => ("ci_lo", "ci_hi"))[SHADE]
const OUT_PNG = joinpath(intervention_output_dir(), "personal_best_worst_change_by_shift$(SHADE == :p95 ? "" : "_$(SHADE)").png")
const OUT_CSV = joinpath(intervention_output_dir(), "personal_best_worst_change_by_shift.csv")
const COL_BASELINE, COL_BEST, COL_WORST = "#000000", "#0072B2", "#D55E00"   # as Fig 5 (Okabe–Ito)
const FILL_ALPHA = 0.30
const S13_H_IN = 6.85   # full page (≤ FULL_PAGE_H_IN); the key strip is just tall enough for the two-row key
const N_BOOT, BOOT_SEED = 2000, 2026_10_13
const FS_TICK, FS_GUIDE, FS_LABEL, FS_LEGEND = 9, 10, 10, 8
const COND_TITLE = Dict("LowLight150lux" => "Low Light (150 lux)", "HighLight10000lux" => "High Light (10,000 lux)")

q(v, p) = quantile(v, p)
"""95 % percentile bootstrap intervals of the median for each vector in `vs` (same participant resamples for all)."""
function median_cis(rng, vs...)
    n = length(first(vs))
    B = [map(v -> median(view(v, idx)), vs) for idx in (rand(rng, 1:n, n) for _ in 1:N_BOOT)]
    return [(quantile(getindex.(B, k), 0.025), quantile(getindex.(B, k), 0.975)) for k in eachindex(vs)]
end
rng = MersenneTwister(BOOT_SEED)
rows = NamedTuple[]
for c in REGULAR_SCHEDULE_CONDITIONS
    fr = JLD2.load(joinpath(intervention_condition_dir(N_PARTICLES_DEFAULT, c), "systematic_rescue_analysis_steadyState.jld2"), "final_results")
    W = fr["analysis_info"]["target_days"]; br, ir = fr["baseline_results"], fr["intervention_results"]
    for tz in sort(collect(keys(br)))
        b = censored_times(br[tz].adaptation_times, W)
        M = permutedims(hcat([censored_times(ir[k].results_by_timezone[tz].intervention_times, W) for k in 1:length(ir)]...))
        dbest = min.(vec(minimum(M, dims=1)), b) .- b   # ≤ 0
        dworst = vec(maximum(M, dims=1)) .- b          # ≥ 0 for nearly everyone
        pbest, pworst = 100 .* dbest ./ b, 100 .* dworst ./ b   # baseline ≥ 1 day for every participant
        ci_db, ci_dw, ci_pb, ci_pw = median_cis(rng, dbest, dworst, pbest, pworst)
        push!(rows, (condition=c, tz=tz, n=length(b),
                     best_change_median=median(dbest), best_change_q25=q(dbest, 0.25), best_change_q75=q(dbest, 0.75),
                     worst_change_median=median(dworst), worst_change_q25=q(dworst, 0.25), worst_change_q75=q(dworst, 0.75),
                     best_change_p025=q(dbest, 0.025), best_change_p975=q(dbest, 0.975),
                     worst_change_p025=q(dworst, 0.025), worst_change_p975=q(dworst, 0.975),
                     best_pct_p025=q(pbest, 0.025), best_pct_p975=q(pbest, 0.975),
                     worst_pct_p025=q(pworst, 0.025), worst_pct_p975=q(pworst, 0.975),
                     best_pct_median=median(pbest), best_pct_q25=q(pbest, 0.25), best_pct_q75=q(pbest, 0.75),
                     worst_pct_median=median(pworst), worst_pct_q25=q(pworst, 0.25), worst_pct_q75=q(pworst, 0.75),
                     best_change_ci_lo=ci_db[1], best_change_ci_hi=ci_db[2], worst_change_ci_lo=ci_dw[1], worst_change_ci_hi=ci_dw[2],
                     best_pct_ci_lo=ci_pb[1], best_pct_ci_hi=ci_pb[2], worst_pct_ci_lo=ci_pw[1], worst_pct_ci_hi=ci_pw[2],
                     n_faster_best=count(<(0), dbest), n_unchanged_best=count(==(0), dbest),
                     n_slower_worst=count(>(0), dworst), n_unchanged_worst=count(==(0), dworst), n_faster_worst=count(<(0), dworst)))
    end
end
df = DataFrame(rows)
CSV.write(OUT_CSV, df)

default(fontfamily=FIG_BOLD, framestyle=:box, dpi=300, grid=true, gridalpha=0.25, gridcolor=:lightgray,
        tickfont=(FIG_BOLD, FS_TICK), guidefont=(FIG_BOLD, FS_GUIDE), legendfont=(FIG_REGULAR, FS_LEGEND),
        titlefont=(FIG_BOLD, FS_LABEL), background_color=:white)
band!(p, x, lo, hi, col) = plot!(p, x, lo; fillrange=hi, fillcolor=col, fillalpha=FILL_ALPHA, linewidth=0, label="")

n_all = first(df.n)
ymin = floor(minimum(df[!, "best_change_$(BAND[1])"]) / 10) * 10 - 5
ymax = ceil(maximum(df[!, "worst_change_$(BAND[2])"]) / 10) * 10 + 5
nmax = 1.45 * n_all   # headroom above the ±all-participants lines for the panel letter
count_ticks = -1000:500:1000
pmax = ceil(maximum(df[!, "worst_pct_$(BAND[2])"]) / 50) * 50 + 25   # % axis: −100 (no time left) to the largest upper CI limit
const ROWS = (:days, :pct, :count)
panels = []
for (row, metric) in enumerate(ROWS), (ci, c) in enumerate(REGULAR_SCHEDULE_CONDITIONS)
    d = sort(df[df.condition .== c, :], :tz)
    left, bottom = ci == 1, row == length(ROWS)
    # y tick labels on the left column only (both columns share each row's scale); no inner margins
    p = plot(xlims=(0, 24), xticks=(4:4:24, bottom ? string.(4:4:24) : fill("", 6)), legend=false,
             xlabel=(bottom ? "Time Zone Shift (Hours)" : ""), title=(row == 1 ? COND_TITLE[c] : ""), titlelocation=:left,
             yformatter=(left ? :auto : (_ -> "")),
             left_margin=(left ? 3Plots.mm : 0Plots.mm), right_margin=(left ? 0Plots.mm : 2Plots.mm), bottom_margin=0Plots.mm,
             top_margin=(row == 1 ? 1Plots.mm : 2Plots.mm))
    if metric in (:days, :pct)
        lo, hi = BAND
        k = metric == :days ? "change" : "pct"
        bw = (d[!, "worst_$(k)_$(lo)"], d[!, "worst_$(k)_$(hi)"], d[!, "worst_$(k)_median"])
        bb = (d[!, "best_$(k)_$(lo)"], d[!, "best_$(k)_$(hi)"], d[!, "best_$(k)_median"])
        band!(p, d.tz, bw[1], bw[2], COL_WORST)
        band!(p, d.tz, bb[1], bb[2], COL_BEST)
        hline!(p, [0]; color=COL_BASELINE, linewidth=2)
        plot!(p, d.tz, bb[3]; color=COL_BEST, linestyle=:dot, linewidth=2.2)
        plot!(p, d.tz, bw[3]; color=COL_WORST, linestyle=:dot, linewidth=2.2)
        if metric == :days
            plot!(p; ylims=(ymin, ymax), ylabel=(left ? "Change in Adaptation\nTime (Days)" : ""))
        else
            plot!(p; ylims=(-100, pmax), yticks=-100:(pmax > 300 ? 200 : 100):pmax, ylabel=(left ? "Change in Adaptation\nTime (% of Baseline)" : ""))
        end
    else
        hline!(p, [-n_all, n_all]; color=:gray50, linestyle=:dash, linewidth=1)
        hline!(p, [0]; color=COL_BASELINE, linewidth=2)
        plot!(p, d.tz, -d.n_faster_best; color=COL_BEST, linestyle=:dot, linewidth=2.2, marker=:circle, markersize=2.5,
              markerstrokewidth=0)
        plot!(p, d.tz, d.n_slower_worst; color=COL_WORST, linestyle=:dot, linewidth=2.2, marker=:circle, markersize=2.5,
              markerstrokewidth=0)
        plot!(p; ylims=(-nmax, nmax), yticks=(count_ticks, left ? string.(abs.(count_ticks)) : fill("", length(count_ticks))),
              ylabel=(left ? "Participants Faster (−)\nor Slower (+)" : ""))
    end
    annotate!(p, 0.9, (metric == :days ? ymax - 0.08 * (ymax - ymin) : metric == :pct ? pmax - 0.08 * (pmax + 100) : 0.88 * nmax),
              text(string('A' + 2 * (row - 1) + (ci - 1)), FIG_BOLD, FS_LABEL + 2, :left))
    push!(panels, p)
end
# Line key below the panels, as in the other figures (inside a panel it would hide data); dashed = all participants.
# Centred under the panels, not the canvas (the left column's axis labels shift the panels ~20 mm right). GR places
# a :top legend well below the strip top, so the strip is pulled up under the x labels (transparent, so it cannot
# cover them).
key = plot(framestyle=:none, showaxis=false, grid=false, xlims=(0, 1), ylims=(0, 1), legend=:top, legend_column=2,
           foreground_color_legend=nothing, background_color_legend=:transparent, margin=0Plots.mm, left_margin=19.5Plots.mm,
           top_margin=-3Plots.mm, background_color_subplot=:transparent)
plot!(key, [-1.0], [0.0]; color=COL_BASELINE, linewidth=2, label="Baseline")
plot!(key, [-1.0], [0.0]; color=COL_BEST, linestyle=:dot, linewidth=2.2, label="Oracle personal best")
plot!(key, [-1.0], [0.0]; color=COL_WORST, linestyle=:dot, linewidth=2.2, label="Oracle personal worst")
plot!(key, [-1.0], [0.0]; color=:gray50, linestyle=:dash, linewidth=1, label="All participants ($(n_all))")
fig = plot(panels..., key; layout=@layout([grid(3, 2); k{0.034h}]), size=print_size(SI_WIDTH_IN, S13_H_IN))
savefig(fig, OUT_PNG)

for c in REGULAR_SCHEDULE_CONDITIONS
    d = df[df.condition .== c, :]
    @printf("%s: median change best %.0f to %.0f d, worst %+.0f to %+.0f d; faster with best %d–%d of %d; slower with worst %d–%d; faster with worst max %d\n",
            c, minimum(d.best_change_median), maximum(d.best_change_median), minimum(d.worst_change_median), maximum(d.worst_change_median),
            minimum(d.n_faster_best), maximum(d.n_faster_best), n_all, minimum(d.n_slower_worst), maximum(d.n_slower_worst), maximum(d.n_faster_worst))
    @printf("    median %% change: best %.0f to %.0f %% (largest at %s h), worst %+.0f to %+.0f %% (largest at %s h)\n",
            minimum(d.best_pct_median), maximum(d.best_pct_median), join(d.tz[d.best_pct_median .== minimum(d.best_pct_median)], ", "),
            minimum(d.worst_pct_median), maximum(d.worst_pct_median), join(d.tz[d.worst_pct_median .== maximum(d.worst_pct_median)], ", "))
end
println("Wrote $OUT_PNG and $OUT_CSV")

# S1 Fig: the virtual participants excluded by the entrainment rule (revision_baseline.jl) and why.
#   A    : µH/χ vs τ for all participants (dotted: prior means, τ = 24.2 h and µH/χ = 4.2/45); excluded participants coloured by what they do over the scoring window
#          (days MLE_START_DAY–BASELINE_PST_DAYS) in the light condition(s) where they fail, marker = where they fail.
#   B–F  : sleep rasters (one row per day, noon to noon, days 56–68; lights off shaded; sleep bars; CBTmin dots) for a typical
#          included participant and one example of each exclusion type.
# The CBTmin example (F) is one that never reaches a steady state (settled_preshift_cbtmin, 200 days).
# Exclusion types, from the histories and the rule each participant failed (revision_baseline.jl):
#   no sleep in the window | sleep on fewer than every night (< 0.75 episodes per day) | CBTmin or sleep midpoint
#   outside the ±1 h band (timing not stable) | a missed dark period with about daily sleep (sleeps in the light period).
# Paper style (figure_style.jl), drawn at print size: SI width (6.5 × 5.6 in), text 7–9 pt.
# Run: julia --project=. experiments/2_RegularScheduleAnalysis/plot_excluded_participants.jl

using CircadianPOMDP
using JLD2
using CSV
using DataFrames
using Statistics
using Plots
using Printf
include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
include(joinpath(@__DIR__, "..", "figure_style.jl"))

const OUT_PNG = normpath(joinpath(regular_schedule_dir(), "..", "excluded_participants.png"))
const CONDS = ("LowLight150lux", "HighLight10000lux")
const LUX = Dict("LowLight150lux" => "150 lux", "HighLight10000lux" => "10,000 lux")
const T0, T1 = 24.0 * MLE_START_DAY, 24.0 * BASELINE_PST_DAYS
const OFF_H, ON_H = CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_OFF_H, CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_ON_H
# Okabe–Ito; included participants in grey
const TYPES = [(:none, "No sleep in 14 days", "#D55E00"), (:subdaily, "Sleeps every second night", "#E69F00"),
               (:daylight, "Sleeps in the light period", "#0072B2"), (:unstable, "Timing not stable", "#CC79A7")]   # CBTmin or sleep midpoint outside the band
const COL_INCLUDED = "#BDBDBD"
const COL_SLEEP, COL_DARK = "#000000", "#E6E6E6"   # raster sleep bars and lights-off shading (CBTmin dots drawn on top)
const SHAPE = Dict(:both => :circle, "LowLight150lux" => :utriangle, "HighLight10000lux" => :rect)

default(fontfamily=FIG_BOLD, framestyle=:box, dpi=300, grid=true, gridalpha=0.2, tickfont=(FIG_BOLD, PT_TICK),
        guidefont=(FIG_BOLD, PT_GUIDE), titlefont=(FIG_BOLD, PT_LABEL), legendfont=(FIG_BOLD, PT_LEGEND),
        background_color=:white)

# ---- data ----
P = load(initial_particles_path(N_PARTICLES_DEFAULT))["initial_particles"]
inc = CSV.read(replace(included_particles_path(), ".jld2" => ".csv"), DataFrame)
reason = Dict(c => Dict(r.particle_id => coalesce(r[Symbol("reason_$c")], "") for r in eachrow(inc)) for c in CONDS)
group(ev) = (d = Dict{Int,Vector{Any}}(); for e in ev; push!(get!(d, e.particle_id, Any[]), e); end; d)
H = Dict(c => (h = load_particle_histories(regular_histories_file(; condition=c));
               (on=group(h.sleep_onset_events), off=group(h.sleep_offset_events), cb=group(h.cbtmin_events))) for c in CONDS)

"""Sleep episodes (start, end) in hours overlapping [lo, hi), for participant `id` in condition `c`."""
function episodes(c, id; lo=T0, hi=T1)
    on = sort(get(H[c].on, id, Any[]); by=e -> e.t_hours); off = sort(get(H[c].off, id, Any[]); by=e -> e.t_hours)
    pairs = CircadianPOMDP._pair_onset_offset_events(on, off; start_t_hours=lo - 24)
    [(p.onset.t_hours, p.offset.t_hours) for p in pairs if p.offset.t_hours > lo && p.onset.t_hours < hi]
end
episodes_per_day(c, id) = count(e -> T0 <= e[1] < T1, episodes(c, id)) / (T1 - T0) * 24

failing(id) = [c for c in CONDS if !isempty(reason[c][id])]
function exclusion_type(id)
    cs = failing(id)
    per_day = minimum(episodes_per_day(c, id) for c in cs)
    per_day == 0 && return :none
    per_day < 0.75 && return :subdaily
    # about daily sleep: either the phase is not stable (CBTmin or sleep-midpoint band) or a dark period was missed
    any(c -> endswith(reason[c][id], "band"), cs) && return :unstable
    all(c -> reason[c][id] == "not asleep every night", cs) && return :daylight
    error("unclassified exclusion for #$id: " * join((reason[c][id] for c in cs), "; "))
end
excluded = sort(inc.particle_id[.!inc.included])
etype = Dict(id => exclusion_type(id) for id in excluded)
where_fails(id) = (cs = failing(id); length(cs) == 2 ? :both : cs[1])
for (t, lab, _) in TYPES
    ids = [id for id in excluded if etype[id] == t]
    println(rpad(lab, 30), length(ids), "  (", join(sort(unique(string.(where_fails.(ids)))), ", "), ")")
end

# ---- A: parameter scatter ----
µH = [p.mu_h for p in P]; µHχ = [p.mu_h / p.chi for p in P]
isinc = Set(inc.particle_id[inc.included])
τv = [p.tau_x for p in P]
pa = plot(xlabel="\\tau (Hours)", ylabel="\\mu_{H}\u2215\\chi", legend=false, xlims=(23.4, 25.0),
          left_margin=3Plots.mm, bottom_margin=2Plots.mm, right_margin=0Plots.mm, top_margin=1Plots.mm)
ii = [i for i in eachindex(P) if i in isinc]
# reference lines (behind the points), the prior means: τ = 24.2 h and µH/χ = 4.2/45 (µH and χ prior means)
vline!(pa, [24.2]; color=:gray45, linestyle=:dot, linewidth=1.2)
hline!(pa, [4.2 / 45]; color=:gray45, linestyle=:dot, linewidth=1.2)
@printf("Panel A reference lines: τ = 24.2 h, µH/χ = 4.2/45 = %.4f (sample mean %.4f)\n", 4.2 / 45, mean(µHχ))
scatter!(pa, τv[ii], µHχ[ii]; markersize=1.6, markerstrokewidth=0, color=COL_INCLUDED, alpha=0.8)
for (t, _, col) in TYPES, w in (:both, CONDS...)
    ids = [id for id in excluded if etype[id] == t && where_fails(id) == w]
    isempty(ids) && continue
    scatter!(pa, τv[ids], µHχ[ids]; markershape=SHAPE[w], markersize=4, color=col, markerstrokecolor=:black,
             markerstrokewidth=0.4)
end
plot!(pa; ylims=(minimum(µHχ) - 0.004, maximum(µHχ) + 0.004))   # the reference lines would otherwise stretch the axis
annotate!(pa, 23.4 + 0.03 * 1.6, ylims(pa)[2] - 0.06 * (ylims(pa)[2] - ylims(pa)[1]), text("A", FIG_BOLD, PT_LABEL + 2, :left))

# Legend panel: colour = type (n), marker = light condition(s) where the participant fails
pl = plot(framestyle=:none, showaxis=false, grid=false, xlims=(0, 1), ylims=(0, 1), legend=:left,
          foreground_color_legend=nothing, background_color_legend=:transparent, legendfont=(FIG_BOLD, PT_LEGEND))
scatter!(pl, [-1], [-1]; color=COL_INCLUDED, markerstrokewidth=0, markersize=3, label="Included (N = $(length(isinc)))")
for (t, lab, col) in TYPES
    n = count(id -> etype[id] == t, excluded)   # `\\ ` (TeX text, for the subscript) must end the label
    scatter!(pl, [-1], [-1]; color=col, markerstrokecolor=:black, markerstrokewidth=0.4, markersize=4,
             label=(endswith(lab, "\\ ") ? "$(lab[1:end-2]) (N = $n)\\ " : "$lab (N = $n)"))
end
for (w, lab) in ((:both, "Fails at both light levels"), ("LowLight150lux", "Fails at 150 lux only"),
                 ("HighLight10000lux", "Fails at 10,000 lux only"))
    scatter!(pl, [-1], [-1]; markershape=SHAPE[w], color=:white, markerstrokecolor=:black, markersize=4, label=lab)
end
plot!(pl, [-1, -1], [-1, -1]; color=:gray45, linestyle=:dot, linewidth=1.5, label="Prior mean (A)")
# Raster key (B–F)
plot!(pl, [-1, -1], [-1, -1]; linewidth=6, color=COL_SLEEP, label="Sleep (B–F)")
scatter!(pl, [-1], [-1]; markershape=:rect, color=COL_DARK, markerstrokecolor=:gray60, markersize=5, label="Lights off (B–F)")
scatter!(pl, [-1], [-1]; color=:white, markerstrokecolor=:black, markersize=4, label="CBT_{MIN} (B–F)\\ ")

# ---- B–F: sleep rasters ----
"""Typical included participant: closest to the included medians of µH and µH/χ (standardized)."""
function typical_included()
    ids = collect(isinc); m1, s1 = median(µH[ids]), std(µH[ids]); m2, s2 = median(µHχ[ids]), std(µHχ[ids])
    ids[argmin([((µH[i] - m1) / s1)^2 + ((µHχ[i] - m2) / s2)^2 for i in ids])]
end
"""Example of an exclusion type: the participant whose behaviour is most typical of its type."""
function example(t)
    ids = [id for id in excluded if etype[id] == t]
    t == :subdaily && return ids[argmin([abs(episodes_per_day(failing(id)[1], id) - 0.5) for id in ids])]
    if t == :unstable   # one that never entrains: no steady state even after the 200-day settled-reference run
        never = [id for id in ids if any(c -> settled_preshift_cbtmin(c; compute=false)[id] === nothing, failing(id))]
        pool = isempty(never) ? ids : never
        return pool[argmax([maximum(cbtmin_dev(c, id) for c in failing(id)) for id in pool])]
    end
    t == :daylight && return ids[argmin([dark_fraction(failing(id)[1], id) for id in ids])]
    return ids[1]
end
function cbtmin_dev(c, id)
    x = [mod(e.t_hours, 24) for e in get(H[c].cb, id, Any[]) if T0 <= e.t_hours < T1]
    isempty(x) ? 0.0 : maximum(abs.(mod.(x .- x[1] .+ 12, 24) .- 12))
end
indark(h) = (h = mod(h, 24); h >= OFF_H || h < ON_H)
function dark_fraction(c, id)
    ep = [(max(s, T0), min(e, T1)) for (s, e) in episodes(c, id)]
    tot = sum(e - s for (s, e) in ep; init=0.0)
    tot == 0 ? 1.0 : sum(count(indark, s:1/12:e-1/12) / 12 for (s, e) in ep) / tot
end

function raster(c, id, title, col; first=false, xlabel="", stroke=:white)
    days = Int(MLE_START_DAY):Int(BASELINE_PST_DAYS) - 2   # noon-to-noon rows whose night ends before the run does
    p = plot(xlims=(12, 36), ylims=(days[1] - 0.5, days[end] + 0.5), yflip=true, legend=false, grid=false,
             xticks=(18:6:30, ["18", "0", "6"]), yticks=(days[1]:4:days[end], first ? string.(days[1]:4:days[end]) : fill("", length(days[1]:4:days[end]))),
             xlabel=xlabel, ylabel=(first ? "Day" : ""), title=title, titlelocation=:left,
             titlefont=(FIG_BOLD, PT_LEGEND), left_margin=(first ? 2Plots.mm : 0Plots.mm), right_margin=1Plots.mm,
             bottom_margin=2Plots.mm)
    # lights off (22:45–06:45), shown on the noon-to-noon axis
    plot!(p, Shape([OFF_H, ON_H + 24, ON_H + 24, OFF_H], [days[1] - 0.5, days[1] - 0.5, days[end] + 0.5, days[end] + 0.5]);
          color=COL_DARK, linewidth=0)
    for d in days, (s, e) in episodes(c, id; lo=24d + 12, hi=24d + 36)
        a, b = max(s, 24d + 12) - 24d, min(e, 24d + 36) - 24d
        plot!(p, Shape([a, b, b, a], [d - 0.32, d - 0.32, d + 0.32, d + 0.32]); color=COL_SLEEP, linewidth=0)
    end
    cbx, cby = Float64[], Float64[]
    for e in get(H[c].cb, id, Any[])
        d = floor(Int, (e.t_hours - 12) / 24)
        d in days && (push!(cbx, e.t_hours - 24d); push!(cby, d))
    end
    scatter!(p, cbx, cby; color=col, markerstrokecolor=stroke, markerstrokewidth=0.8, markersize=3.8)
    return p
end
idT = typical_included()
examples = [(idT, "LowLight150lux", "Included", "#FFFFFF")]   # white CBTmin dots (black edge) for the included example
for (t, lab, col) in TYPES
    id = example(t); c = failing(id)[1]
    push!(examples, (id, c, replace(lab, "\\ " => "", "CBT_{MIN}" => "CBT_{MIN}"), col))
end
letters = "BCDEF"
const SHORT = ["Included", "No sleep", "Every 2nd night", "Daytime sleep", "Not entrained"]
rasters = [raster(c, id, "$(letters[k])  $(SHORT[k])\n    $(LUX[c]), #$id", col; first=(k == 1),
                  xlabel=(k == 3 ? "Clock Time (Hours)" : ""), stroke=(k == 1 ? :black : :white)) for (k, (id, c, _, col)) in enumerate(examples)]
for (id, c, lab, _) in examples
    @printf("example %-28s #%-4d %s: %.2f episodes/day, %.0f %% of sleep in the dark, CBTmin moved %.1f h\n",
            lab, id, LUX[c], episodes_per_day(c, id), 100dark_fraction(c, id), cbtmin_dev(c, id))
end

fig = plot(plot(pa, pl; layout=grid(1, 2; widths=[0.59, 0.41])), plot(rasters...; layout=(1, 5));
           layout=grid(2, 1; heights=[0.52, 0.48]), size=print_size(SI_WIDTH_IN, 5.6))
savefig(fig, OUT_PNG)
println("Wrote $OUT_PNG")

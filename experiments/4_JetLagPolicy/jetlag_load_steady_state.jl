# Shared loading logic for Jet Lag steady-state analysis.
# Used by plot_baseline_vs_intervention_wrapped_quartiles_steadyState.jl (Fig 5), plot_personal_best_worst_change.jl
# (S14 Fig).
using CircadianPOMDP
using JLD2
using Statistics

# Include reference for paths (must be in same dir or parent)
const _JETLAG_LOAD_DIR = @__DIR__
include(joinpath(_JETLAG_LOAD_DIR, "..", "jetlag_reference.jl"))

# Paths
const ADAPTATIONS_BASE = intervention_adaptations_dir()
const ADAPTATIONS_BASES = [
    ADAPTATIONS_BASE,
    joinpath(JETLAG_DATA_ROOT, "4_JetLagPolicy", "Adaptations"),
]
const PARTICLES_DEFAULT = N_PARTICLES_DEFAULT

_get(obj, sym::Symbol) = obj isa NamedTuple ? getproperty(obj, sym) : get(obj, string(sym), get(obj, sym, nothing))

function find_steady_state_results_path(condition_name::String; particles::Int=PARTICLES_DEFAULT, mini::Bool=false)
    for adaptations_base in ADAPTATIONS_BASES
        if mini
            base = intervention_condition_dir(particles, condition_name; mini=true)
        else
            base = intervention_condition_dir(particles, condition_name)
        end
        path = joinpath(base, "systematic_rescue_analysis_steadyState.jld2")
        if isfile(path)
            return path
        end
    end
    return nothing
end

function load_from_checkpoints(dir::String)
    baseline_path = joinpath(dir, "baseline_results_steadyState.jld2")
    cumulative_path = joinpath(dir, "intervention_results_cumulative_steadyState.jld2")
    if !isfile(baseline_path)
        return nothing
    end
    baseline_data = load(baseline_path)
    baseline_results = baseline_data["data"]
    intervention_results = isfile(cumulative_path) ? load(cumulative_path)["data"] : Dict()
    for f in readdir(dir)
        if startswith(f, "policy_") && endswith(f, "_results_steadyState.jld2")
            d = load(joinpath(dir, f))
            inner = haskey(d, "data") ? d["data"] : d
            idx = inner["policy_idx"]
            intervention_results[idx] = (policy=inner["policy"], results_by_timezone=inner["results_by_timezone"])
        end
    end
    timezones = sort(collect(keys(baseline_results)))
    best_policies_by_timezone = Dict{Int, Vector{Any}}()
    for tz_shift in timezones
        policy_performances = Any[]
        for (policy_idx, policy_data) in intervention_results
            rbt = _get(policy_data, :results_by_timezone)
            haskey(rbt, tz_shift) || continue
            r = rbt[tz_shift]
            diff = _get(r, :adaptation_time_difference)
            push!(policy_performances, (policy_idx=policy_idx, policy=_get(policy_data, :policy), adaptation_time_difference=diff,
                baseline_median_adaptation=_get(r, :baseline_median_adaptation), intervention_median_adaptation=_get(r, :intervention_median_adaptation),
                baseline_sample_size=_get(r, :baseline_sample_size), intervention_sample_size=_get(r, :intervention_sample_size)))
        end
        sort!(policy_performances, by=x->x.adaptation_time_difference, rev=false)
        best_policies_by_timezone[tz_shift] = policy_performances
    end
    return baseline_results, intervention_results, best_policies_by_timezone
end

# ---- Right-censoring at the end of the post-shift window ----
# Adaptation can only be detected within the run's post-shift window (`window_days`: analysis_info["target_days"],
# INTERVENTION_TARGET_DAYS), so an included participant with no steady state by then adapted later (every included
# participant has a baseline CBTmin, so a missing time always means this). Summaries count these participants as
# `window_days`, i.e. ranked slowest: the median and quartiles are then exact while fewer than 25 % are censored.
# Dropping them understates adaptation time (with a 50-day window, medians were 1–2 d too low at 150 lux).
# Participants outside the shared set are left out entirely.
const INCLUDED_IDS_SORTED = included_particle_ids()

"""Adaptation times of the included participants (id order), non-adapters counted as `window_days`."""
censored_times(times, window_days) =
    Float64[isfinite(times[i]) ? Float64(times[i]) : Float64(window_days) for i in INCLUDED_IDS_SORTED]

"""Percent of included participants not adapted within the post-shift window."""
pct_not_adapted(times) = 100 * count(i -> !isfinite(times[i]), INCLUDED_IDS_SORTED) / length(INCLUDED_IDS_SORTED)

"""
Ranking key of a candidate general policy (lower is better), from the participants' censored times under the policy
(`t`) and without intervention (`b`), paired by participant: most participants helped (strictly fewer days; adapting
within the window when the baseline does not counts), then fewest harmed, then the larger median personal
improvement. A median of times can favour a policy that speeds up the middle of the distribution while slowing most
individuals (seen at 150 lux, 12 h, when the population was reweighted), so the rule counts people instead.
"""
general_policy_key(t, b) = (-count(t .< b), count(t .> b), -median(b .- t))

"""Mirror of `general_policy_key` for the general worst: most participants harmed, then fewest helped, then the
largest median personal loss."""
general_worst_key(t, b) = (-count(t .> b), count(t .< b), median(b .- t))

"""
General (population) policy for `tz_shift`, chosen by `key` (default `general_policy_key`; `general_worst_key` for
the general worst) against the no-intervention times `baseline_times` (ties: lower policy index). Returns `nothing` if
no policy has the shift; `n_tied` counts policies tied on the first criterion.
"""
function select_general_policy(intervention_results, tz_shift, window_days, baseline_times; key=general_policy_key)
    b = censored_times(baseline_times, window_days)
    candidates = Any[]
    for (policy_idx, policy_data) in intervention_results
        rbt = _get(policy_data, :results_by_timezone)
        haskey(rbt, tz_shift) || continue
        t = censored_times(_get(rbt[tz_shift], :intervention_times), window_days)
        push!(candidates, (key=(key(t, b)..., Int(policy_idx)), policy_idx=policy_idx,
                           policy=_get(policy_data, :policy), times=_get(rbt[tz_shift], :intervention_times),
                           pct_helped=100 * mean(t .< b), pct_harmed=100 * mean(t .> b)))
    end
    isempty(candidates) && return nothing
    best = candidates[argmin([c.key for c in candidates])]
    n_tied = count(c -> c.key[1] == best.key[1], candidates)
    return (policy_idx=best.policy_idx, policy=best.policy, times=best.times, n_tied=n_tied,
            pct_helped=best.pct_helped, pct_harmed=best.pct_harmed)
end

"""
Load one condition from unified steady-state JLD2 (or from checkpoints).
If return_raw_times=true, also returns (general_best_raw, personalized_raw, baseline_raw): per timezone, the
included participants' times in id order with non-adapters at the window length (so the three are participant-aligned).
If return_details=true, a final NamedTuple of per-timezone vectors is appended: the general policy, how many
policies tie with it, the percent of participants not adapted within the window for each strategy, and `window_days`.

All summaries count non-adapters as the post-shift window length (see `censored_times`).

**General** (population, non-oracle): per timezone, the single policy chosen by `select_general_policy`.

**Personalized**: per participant and timezone, the best time over all policies and no intervention (window length
if nothing adapts them within the window); then median and band over participants. Every policy replaces the first
day in the new time zone with a dark intervention day, which can slow a very slow adapter (one case in 44,114:
150 lux, 8 h, 56 vs 57 days), so no intervention is one of the oracle's options.

**Oracle worst** (`worst_*` in `details`): the oracle's counterpart, each participant's longest time over the light windows
(no intervention not offered: it measures how much a mistimed window can cost).

`ci_quantiles`: `(low, high)` percentiles for the shaded band (default `(0.25, 0.75)` = IQR).
Use `(0.025, 0.975)` for a central 95% empirical interval (not a bootstrap CI for the median); quantiles above
the censored fraction's complement are not exact.
"""
function load_unified_steady_state_data(
    condition_name::String;
    particles::Int=PARTICLES_DEFAULT,
    return_raw_times::Bool=false,
    return_details::Bool=false,
    ci_quantiles::Tuple{Float64,Float64}=(0.25, 0.75),
)
    @assert 0.0 < ci_quantiles[1] < ci_quantiles[2] < 1.0
    jld2_path = find_steady_state_results_path(condition_name; particles=particles, mini=false)
    baseline_results = nothing
    intervention_results = nothing
    best_policies_by_timezone = nothing
    window_days = intervention_target_days(condition_name)

    if jld2_path !== nothing
        data = load(jld2_path)
        final_results = data["final_results"]
        baseline_results = final_results["baseline_results"]
        intervention_results = final_results["intervention_results"]
        best_policies_by_timezone = final_results["best_policies_by_timezone"]
        window_days = get(final_results["analysis_info"], "target_days", window_days)
        println("   ✓ Loaded from $(jld2_path) (post-shift window $(window_days) d)")
    else
        for adaptations_base in ADAPTATIONS_BASES
            dir = intervention_condition_dir(particles, condition_name)
            if isdir(dir)
                loaded = load_from_checkpoints(dir)
                if loaded !== nothing
                    baseline_results, intervention_results, best_policies_by_timezone = loaded
                    println("   ✓ Loaded from checkpoints in $(dir)")
                    break
                end
            end
        end
    end

    if baseline_results === nothing
        error("Unified steady-state results not found in $(intervention_condition_dir(particles, condition_name)). " *
              "Run rescuingJetLag_unified.jl (or pull the Sherlock results) first.")
    end

    timezones = sort(collect(keys(baseline_results)))
    q_lo, q_hi = ci_quantiles
    summarize(t) = (median(t), quantile(t, q_lo), quantile(t, q_hi))

    general_best_raw = Dict{Int, Vector{Float64}}()
    personalized_raw = Dict{Int, Vector{Float64}}()
    baseline_raw = Dict{Int, Vector{Float64}}()
    baseline_times, baseline_ci_low, baseline_ci_high = Float64[], Float64[], Float64[]
    intervention_times, intervention_ci_low, intervention_ci_high = Float64[], Float64[], Float64[]
    personalized_times, personalized_ci_low, personalized_ci_high = Float64[], Float64[], Float64[]
    details = (general_policy_idx=Any[], general_policy=Any[], n_tied=Int[], pct_helped_general=Float64[],
               pct_harmed_general=Float64[], worst_median=Float64[], worst_q25=Float64[], worst_q75=Float64[],
               pct_slowed_worst=Float64[], general_worst_policy=Any[], general_worst_median=Float64[],
               general_worst_q25=Float64[], general_worst_q75=Float64[], pct_slowed_general_worst=Float64[],
               pct_not_adapted_baseline=Float64[],
               pct_not_adapted_general=Float64[], pct_not_adapted_personalized=Float64[], window_days=window_days)

    for tz_shift in timezones
        raw_b = _get(baseline_results[tz_shift], :adaptation_times)
        b = censored_times(raw_b, window_days)
        baseline_raw[tz_shift] = b
        m, lo, hi = summarize(b)
        push!(baseline_times, m); push!(baseline_ci_low, lo); push!(baseline_ci_high, hi)

        g = select_general_policy(intervention_results, tz_shift, window_days, raw_b)
        if g === nothing
            push!(intervention_times, NaN); push!(intervention_ci_low, NaN); push!(intervention_ci_high, NaN)
            push!(personalized_times, NaN); push!(personalized_ci_low, NaN); push!(personalized_ci_high, NaN)
            for v in (details.general_policy_idx, details.general_policy); push!(v, nothing); end
            push!(details.n_tied, 0); push!(details.pct_helped_general, NaN); push!(details.pct_harmed_general, NaN)
            for v in (details.worst_median, details.worst_q25, details.worst_q75, details.pct_slowed_worst,
                      details.general_worst_median, details.general_worst_q25, details.general_worst_q75,
                      details.pct_slowed_general_worst); push!(v, NaN); end
            push!(details.general_worst_policy, nothing)
            push!(details.pct_not_adapted_baseline, pct_not_adapted(raw_b))
            push!(details.pct_not_adapted_general, NaN); push!(details.pct_not_adapted_personalized, NaN)
            continue
        end
        gt = censored_times(g.times, window_days)
        general_best_raw[tz_shift] = gt
        m, lo, hi = summarize(gt)
        push!(intervention_times, m); push!(intervention_ci_low, lo); push!(intervention_ci_high, hi)

        # Personalized: each participant's best time over every policy and no intervention (participant-aligned)
        per_policy = [censored_times(_get(_get(pd, :results_by_timezone)[tz_shift], :intervention_times), window_days)
                      for pd in values(intervention_results) if haskey(_get(pd, :results_by_timezone), tz_shift)]
        pt = min.(reduce((a, x) -> min.(a, x), per_policy), b)
        personalized_raw[tz_shift] = pt
        m, lo, hi = summarize(pt)
        push!(personalized_times, m); push!(personalized_ci_low, lo); push!(personalized_ci_high, hi)
        # Oracle worst: each participant's longest time over the light windows
        wt = reduce((a, x) -> max.(a, x), per_policy)
        m, lo, hi = summarize(wt)
        push!(details.worst_median, m); push!(details.worst_q25, lo); push!(details.worst_q75, hi)
        push!(details.pct_slowed_worst, 100 * mean(wt .> b))
        # General worst: the single window that slows the most participants (mirror of the general rule)
        gw = select_general_policy(intervention_results, tz_shift, window_days, raw_b; key=general_worst_key)
        gwt = censored_times(gw.times, window_days)
        m, lo, hi = summarize(gwt)
        push!(details.general_worst_policy, gw.policy)
        push!(details.general_worst_median, m); push!(details.general_worst_q25, lo); push!(details.general_worst_q75, hi)
        push!(details.pct_slowed_general_worst, gw.pct_harmed)

        push!(details.general_policy_idx, g.policy_idx); push!(details.general_policy, g.policy)
        push!(details.n_tied, g.n_tied)
        push!(details.pct_helped_general, g.pct_helped); push!(details.pct_harmed_general, g.pct_harmed)
        push!(details.pct_not_adapted_baseline, pct_not_adapted(raw_b))
        push!(details.pct_not_adapted_general, pct_not_adapted(g.times))
        push!(details.pct_not_adapted_personalized, 100 * count(==(window_days), pt) / length(pt))
    end

    ret = (timezones, baseline_times, baseline_ci_low, baseline_ci_high,
           intervention_times, intervention_ci_low, intervention_ci_high,
           personalized_times, personalized_ci_low, personalized_ci_high)
    return_raw_times && (ret = (ret..., general_best_raw, personalized_raw, baseline_raw))
    return_details && (ret = (ret..., details))
    return ret
end

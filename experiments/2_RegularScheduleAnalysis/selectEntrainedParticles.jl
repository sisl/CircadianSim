# Select the particles included in every analysis (revision).
# Applies the entrainment rule (revision_baseline.jl) to the regular-schedule histories of both light
# conditions over the scoring window (days MLE_START_DAY–BASELINE_PST_DAYS). A particle is included only if
# it is entrained and has an MLE summary in every condition.
#
# Pipeline: generateRegularScheduleParticleHistories.jl → regularSchedule_MLE_particleSummary.jl (both
# conditions) → this script → all downstream analyses (load with included_particle_ids()).
#
# Writes included_particles_path() (JLD2: included_ids, per-particle table) and a CSV next to it.

using JLD2
using CSV
using DataFrames
using CircadianPOMDP

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

println("Selecting entrained particles: days $(MLE_START_DAY)–$(BASELINE_PST_DAYS), " *
        "band ±$(ENTRAINMENT_BAND_H) h, lights off $(round(CircadianPOMDP.REGULAR_SCHEDULE_LIGHT_OFF_H, digits=3)) h")

countmap_reasons(v) = (d = Dict{String,Int}(); for x in v; isempty(x) || (d[x] = get(d, x, 0) + 1); end; d)

rows = DataFrame()
n_particles = 0
for condition in REGULAR_SCHEDULE_CONDITIONS
    hist = load_particle_histories(regular_histories_file(; condition))
    global n_particles = length(hist.particles)
    events = (hist.cbtmin_events, hist.sleep_onset_events, hist.sleep_offset_events)
    metrics = entrainment_metrics(events, n_particles)
    with_summary = Set(s.particle_id for s in load_particle_summaries(regular_summaries_path(; condition)))
    for m in metrics
        push!(rows, (condition=condition, m..., has_summary=m.particle_id in with_summary,
                     reason=exclusion_reason(m, m.particle_id in with_summary)))
    end
    g = rows[rows.condition .== condition, :]
    println("  $condition: fails $(count(!isempty, g.reason)) of $n_particles")
    for (r, k) in sort(collect(countmap_reasons(g.reason)); by=last, rev=true)
        println("    $(rpad(r, 36)) $k")
    end
end

excluded = Set(rows.particle_id[.!isempty.(rows.reason)])
included_ids = sort(setdiff(collect(1:n_particles), excluded))
per_particle = DataFrame(particle_id=1:n_particles)
for condition in REGULAR_SCHEDULE_CONDITIONS
    g = sort(rows[rows.condition .== condition, :], :particle_id)
    per_particle[!, Symbol("reason_", condition)] = g.reason
end
per_particle.included = [id in Set(included_ids) for id in per_particle.particle_id]

out = included_particles_path()
mkpath(dirname(out))
jldsave(out; included_ids, per_particle, metrics=rows, baseline_days=BASELINE_PST_DAYS,
        score_start_day=MLE_START_DAY, band_h=ENTRAINMENT_BAND_H, night_sleep_min_fraction=NIGHT_SLEEP_MIN_FRACTION)
CSV.write(replace(out, ".jld2" => ".csv"), per_particle)
println("Included in every analysis: $(length(included_ids)) of $n_particles " *
        "(excluded $(length(excluded)); fail in both conditions: " *
        "$(count(r -> all(!isempty, r), eachrow(per_particle[:, 2:end-1]))))")
println("Saved $out")

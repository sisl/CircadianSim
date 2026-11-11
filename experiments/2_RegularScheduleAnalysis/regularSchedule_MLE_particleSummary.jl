# Compute particle summaries for the regular schedule.
# Use last 14 days of particle histories (start_day=20.333) to compute MLEs.
# Output is used by compareWithNHANES.jl. Run generateRegularScheduleParticleHistories.jl first.
# Uses CircadianPOMDP package and events-based format from simulate_particle_population_events_only.
#
# Paths and constants from jetlag_reference.jl (outputs to Data/JetLag/2_RegularScheduleAnalysis/particleSummaries/).

using JLD2
using CircadianPOMDP

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

ensure_jetlag_data_dirs()

# Optional: condition from ARGS (e.g. julia script.jl LowLight150lux). Default: HighLight10000lux.
condition = (length(ARGS) >= 1 && ARGS[1] in REGULAR_SCHEDULE_CONDITIONS) ? ARGS[1] : REGULAR_SCHEDULE_CONDITION_DEFAULT
println("Light diet condition: $condition")

# Load events-based format (from generateRegularScheduleParticleHistories.jl)
println("Loading regular schedule events... ")
data_path = regular_histories_file(; condition)
hist = load_particle_histories(data_path)
cbtmin_events = hist.cbtmin_events
sleep_onset_events = hist.sleep_onset_events
sleep_offset_events = hist.sleep_offset_events
particles = hist.particles

# Compute MLE summaries from events (settings from jetlag_reference.jl)
println("Computing MLE summaries from events...")
summaries = compute_mle_population_summaries(
    cbtmin_events, sleep_onset_events, sleep_offset_events, particles;
    dt=DT_HOURS, start_day=MLE_START_DAY, min_sleep_episodes=MLE_MIN_SLEEP_EPISODES,
    time_shift=MLE_TIME_SHIFT, verbose=false
)

# Save summaries (path from jetlag_reference.jl; matches compareWithNHANES.jl)
println("Saving summaries... ")
filepath = regular_summaries_path(; condition)
mkpath(dirname(filepath))
save_mle_summaries(summaries, filepath; include_analysis=false, verbose=false)
println("Done. Saved to $(filepath)")


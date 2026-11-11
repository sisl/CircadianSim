# ================================================================
# Jet Lag Experiment Reference: Paths and Settings
# ================================================================
# Single source of truth for experiments sub-experiments.
# Include from any script under experiments/ as:
#   From 1_GenerateInitialParticles/, 2_RegularScheduleAnalysis/, 3_JetLagAnalysis/, 4_JetLagPolicy/:
#     include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))
#
# Data layout: repo Data/ contains JetLag/ with all experiment outputs. Run ensure_jetlag_data_dirs()
# once to create the folder structure. NHANES and other shared data stay under Data/ (use REPO_DATA_ROOT).
#
# Migration: If you have existing data under Data/1_InitialParticles, Data/2_RegularScheduleAnalysis, etc.,
# move those folders into Data/JetLag/ (e.g. mv Data/1_InitialParticles Data/JetLag/) so paths match.
#
# Reproducibility: Each pipeline script calls Random.seed!() before computation.
# Single-threaded runs (julia script.jl) are fully deterministic.
# Multi-threaded runs (julia -t auto script.jl) are statistically equivalent but not
# bit-for-bit identical, because Threads.@threads task scheduling is non-deterministic.
# Lara Weed, February 2025
# ================================================================

using Dates  # for START_DATE (DateTime)

# Root of the experiments folder (where this file lives: experiments/)
const JETLAG_ROOT = @__DIR__
# Repo Data directory (repo/Data); shared inputs (e.g. NHANES) live here
const REPO_DATA_ROOT = normpath(joinpath(JETLAG_ROOT, "..", "Data"))
# All jet lag experiment outputs live under Data/JetLag/
const JETLAG_DATA_ROOT = joinpath(REPO_DATA_ROOT, "JetLag")

# Filename suffix for V4 alignment (e.g. V4_1.341hEarlier; uses JETLAG_MEAN_ALIGNMENT_SHIFT_H). Requires CircadianPOMDP in scope.
const JETLAG_ALIGNMENT_FILENAME_SUFFIX = "V4_$(JETLAG_MEAN_ALIGNMENT_SHIFT_H)hEarlier"

# ================================================================
# DIRECTORY NAMES (under Data/JetLag/)
# ================================================================

const DIR_INITIAL_PARTICLES = "1_InitialParticles"
const DIR_REGULAR_SCHEDULE = "2_RegularScheduleAnalysis"
const DIR_JETLAG_ANALYSIS = "3_JetLagAnalysis"
const DIR_JETLAG_INTERVENTION = "4_JetLagIntervention"
const SUBDIR_PARTICLE_HISTORIES = "particleHistories"
const SUBDIR_PARTICLE_SUMMARIES = "particleSummaries"
const SUBDIR_CROSS_CORRELATIONS = "crossCorrelations"
const SUBDIR_SHAP = "SHAP"
const SUBDIR_STEADY_STATE = "SteadyState"
const SUBDIR_TRAJECTORIES = "trajectories"
const SUBDIR_LIGHT_DIETS = "LightDiets"
const SUBDIR_ADAPTATIONS = "Adaptations"
const SUBDIR_MINI = "Mini"

# ================================================================
# STANDARD FILE NAMES (single place for clarity; used in path helpers below)
# ================================================================

const FNAME_INITIAL_PARTICLES = "initial_particles"   # use with "\$(N).jld2"
"""Regular-schedule histories filename (length set by BASELINE_PST_DAYS)."""
fname_regular_histories() = "particleHist_regular_$(BASELINE_PST_DAYS)days_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.jld2"
const FNAME_REGULAR_SUMMARIES = "particleSummaries_regular_14days_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.jld2"
const FNAME_STEADY_STATE_RESULTS = "steady_state_detection_results.jld2"
const FNAME_SYSTEMATIC_RESCUE = "systematic_rescue_analysis_steadyState.jld2"

# ================================================================
# INPUT PATHS (read by jet lag scripts)
# ================================================================

"""Path to initial particles JLD2 (e.g. initial_particles1000.jld2)."""
initial_particles_path(N::Int=1000) = joinpath(JETLAG_DATA_ROOT, DIR_INITIAL_PARTICLES, "$(FNAME_INITIAL_PARTICLES)$(N).jld2")

# Regular schedule light diet conditions (subdirs under 2_RegularScheduleAnalysis).
# 150 lux is the manuscript default; 10k lux is the secondary (bright indoor) condition.
const REGULAR_SCHEDULE_CONDITION_HIGHLIGHT = "HighLight10000lux"   # 10k lux
const REGULAR_SCHEDULE_CONDITION_LOWLIGHT = "LowLight150lux"       # 150 lux
const REGULAR_SCHEDULE_CONDITION_DEFAULT = REGULAR_SCHEDULE_CONDITION_LOWLIGHT
const REGULAR_SCHEDULE_CONDITIONS = (REGULAR_SCHEDULE_CONDITION_LOWLIGHT, REGULAR_SCHEDULE_CONDITION_HIGHLIGHT)

"""Directory for regular-schedule analysis outputs. condition: LowLight150lux (default) or HighLight10000lux."""
function regular_schedule_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(JETLAG_DATA_ROOT, DIR_REGULAR_SCHEDULE, condition)
end

"""Directory where regular-schedule particle histories are saved."""
function regular_schedule_particle_hist_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_dir(; condition), SUBDIR_PARTICLE_HISTORIES)
end

"""Directory for particle summaries and NHANES comparison (MLE summary JLD2, particle vs NHANES figure/stats)."""
function regular_schedule_particle_summaries_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_dir(; condition), SUBDIR_PARTICLE_SUMMARIES)
end

"""Directory for cross-correlation outputs (heatmaps, LaTeX table, scatter plots)."""
function regular_schedule_cross_correlations_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_dir(; condition), SUBDIR_CROSS_CORRELATIONS)
end

"""Directory for SHAP analysis outputs (beehive plot, etc.)."""
function regular_schedule_shap_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_dir(; condition), SUBDIR_SHAP)
end

"""Path to regular-schedule particle histories (BASELINE_PST_DAYS long; used for baseline CBTmin)."""
function regular_histories_file(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_particle_hist_dir(; condition), fname_regular_histories())
end

"""Path to regular-schedule particle summaries (14-day MLE)."""
function regular_summaries_path(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(regular_schedule_particle_summaries_dir(; condition), FNAME_REGULAR_SUMMARIES)
end

# ================================================================
# 3_JetLagAnalysis PATHS (baseline jet lag, no intervention)
# ================================================================
# Condition subdirs (HighLight10000lux, LowLight150lux) match 2_RegularScheduleAnalysis.
# Each condition uses its matching regular-schedule baseline (same light diet).

"""Output directory for 3_JetLagAnalysis for a given light condition."""
function jetlag_analysis_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(JETLAG_DATA_ROOT, DIR_JETLAG_ANALYSIS, condition)
end

"""Verbose output from simulate_particle_population_events_only in generateJetLagParticleHistories.jl."""
const JETLAG_VERBOSE = true

"""Particle histories for 3_JetLagAnalysis (generateJetLagParticleHistories.jl output)."""
function jetlag_analysis_particle_hist_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(jetlag_analysis_dir(; condition), SUBDIR_PARTICLE_HISTORIES)
end

"""Steady-state detection results (detectSteadyState.jl output)."""
function steady_state_results_path(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(jetlag_analysis_dir(; condition), SUBDIR_STEADY_STATE, FNAME_STEADY_STATE_RESULTS)
end

"""Trajectory plots (plotJetLagTrajectories.jl output)."""
function jetlag_analysis_trajectories_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(jetlag_analysis_dir(; condition), SUBDIR_TRAJECTORIES)
end

"""Mean |SHAP| by time-zone shift (summarizeAdaptationStatistics.jl writes, plotCombinedLightConditions.jl reads).
v2: rows joined to adaptation times by particle_id (issue #27). v3: settled-phase target, per-condition window, and
participants not adapted within the window included at the window length (censored). `target`: the fitted outcome,
"logdays" (log(1 + days), v4; the default, see ADAPTATION_SHAP_TARGET) or "days" (raw days, v3).
`variant`: "all" (5 static parameters + 5 regular-schedule outputs, as in the manuscript), "parameters"
(static parameters only) or "outputs" (regular-schedule timing only); see computeAdaptationSHAPVariants.jl."""
function jetlag_shap_cache_path(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT, variant::String="all",
                                target::String=ADAPTATION_SHAP_TARGET)
    stem = Dict("logdays" => "shap_importance_cache_v4_logdays", "days" => "shap_importance_cache_v3_censored")[target]
    joinpath(jetlag_analysis_dir(; condition), variant == "all" ? "$(stem).jld2" : "$(stem)_$(variant).jld2")
end

# Outcome of the adaptation-time SHAP / Random Forest fits (Fig 4C–D): log(1 + days) by default, because a few
# participants per shift (1–11 % at 150 lux) take 50–100 days; their exact durations are unpredictable from the
# parameters yet dominate raw-day variance (parameters-only OOF R² 0.48 raw vs 0.71 log at 150 lux, same top
# parameter at every shift). SHAP_TARGET=days gives the raw-day fits (supplementary check).
const ADAPTATION_SHAP_TARGET = get(ENV, "SHAP_TARGET", "logdays")
ADAPTATION_SHAP_TARGET in ("logdays", "days") || error("SHAP_TARGET must be logdays or days")
"""Fitted outcome for adaptation days under `target`."""
adaptation_shap_y(days; target::String=ADAPTATION_SHAP_TARGET) = target == "logdays" ? log1p.(Float64.(days)) : Float64.(days)
"""Out-of-fold R² of the adaptation-time fits per feature set (computeAdaptationSHAPVariants.jl)."""
adaptation_r2_csv_path(; target::String=ADAPTATION_SHAP_TARGET) =
    joinpath(JETLAG_DATA_ROOT, DIR_JETLAG_ANALYSIS, "adaptation_r2_by_feature_set_$(target).csv")

"""SHAP analysis outputs for 3_JetLagAnalysis (heatmaps, trends, beehive plots)."""
function jetlag_analysis_shap_dir(; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT)
    joinpath(jetlag_analysis_dir(; condition), SUBDIR_SHAP)
end

# ================================================================
# 4_JetLagIntervention PATHS (rescue policy, unified pipeline)
# ================================================================

"""Base directory for intervention data (particle histories + Adaptations)."""
intervention_base_dir() = joinpath(JETLAG_DATA_ROOT, DIR_JETLAG_INTERVENTION)

"""Particle histories for intervention runs (full or mini)."""
function intervention_particle_hist_dir(; mini::Bool=false)
    base = intervention_base_dir()
    sub = "$(SUBDIR_PARTICLE_HISTORIES)_$(CBTMIN_DEFINITION_TAG)"   # event caches depend on the CBTmin definition
    return mini ? joinpath(base, SUBDIR_MINI, sub) : joinpath(base, sub)
end

"""Adaptations results base (systematic rescue analysis, checkpoints)."""
intervention_adaptations_dir() = joinpath(intervention_base_dir(), SUBDIR_ADAPTATIONS)

"""Intervention-window version in Exp 4 output names: windows are contiguous blocks on the simulation clock
(`intervention_day_diet` in rescuingJetLag_unified.jl). Earlier runs split the windows starting at 0 or 1 h in two."""
const INTERVENTION_WINDOW_TAG = "contigWindows"

"""Adaptation-target version in Exp 4 output names: settled pre-shift CBTmin + shift (settled_preshift_cbtmin in
revision_baseline.jl). Earlier runs used the day-56–70 CBTmin."""
const TARGET_REFERENCE_TAG = "settledTarget"

"""CBTmin-definition version in Exp 4 output names and in the cached settled reference: CBTmin = time at which
atan(xc, x) reaches −170.7° (St. Hilaire et al. 2007), plus 0.97 h (`cbtmin_phase_crossing`). Results without this tag
were made before the 2026-10 fix, when CBTmin was timed at the ±180° wrap (~0.63 h later)."""
const CBTMIN_DEFINITION_TAG = "cbtStHilaire"

"""Post-shift window (days) of experiments 3 and 4 per light condition. At 150 lux slow re-entrainers need up to
~95 days (no intervention); at 10,000 lux > 99 % of cases adapt within 50 days."""
const INTERVENTION_TARGET_DAYS = Dict("LowLight150lux" => 100, "HighLight10000lux" => 50)
intervention_target_days(condition_name::String) = INTERVENTION_TARGET_DAYS[condition_name]
"""Total jet-lag simulation days (pre-shift + post-shift window) for a condition."""
jetlag_total_days(condition_name::String) = BASELINE_PST_DAYS + intervention_target_days(condition_name)

"""Directory for one condition (e.g. HighLight10000lux) under Adaptations. Versioned by baseline length, alignment,
intervention-window definition, adaptation target and post-shift window, so a run never resumes from checkpoints
made with different settings."""
function intervention_condition_dir(n_particles::Int, condition_name::String; mini::Bool=false,
                                    target_days::Int=(mini ? TARGET_DAYS_MINI : intervention_target_days(condition_name)))
    base = intervention_adaptations_dir()
    run = "Particles$(n_particles)_$(BASELINE_PST_DAYS)d_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_$(INTERVENTION_WINDOW_TAG)_" *
          "$(TARGET_REFERENCE_TAG)_$(CBTMIN_DEFINITION_TAG)_post$(target_days)d"
    return mini ? joinpath(base, SUBDIR_MINI, run, condition_name) : joinpath(base, run, condition_name)
end

"""Path to systematic rescue analysis JLD2 for a condition."""
function systematic_rescue_analysis_path(n_particles::Int, condition_name::String; mini::Bool=false)
    return joinpath(intervention_condition_dir(n_particles, condition_name; mini), FNAME_SYSTEMATIC_RESCUE)
end

# ================================================================
# SHARED DATA (outside JetLag/; use REPO_DATA_ROOT)
# ================================================================

"""Directory for NHANES data (e.g. sleep questionnaire); under repo Data/, not under JetLag/."""
nhanes_data_dir() = joinpath(REPO_DATA_ROOT, "NHANES")

"""Path to NHANES sleep questionnaire file (.xpt or .csv)."""
nhanes_sleep_file() = joinpath(nhanes_data_dir(), "SLQ_L_2021_2023.xpt")

# ================================================================
# CREATE JETLAG DATA DIRECTORY STRUCTURE
# ================================================================

"""
Create Data/JetLag/ and standard subdirectories so scripts have a known layout.
Run once (e.g. from a setup script or the first time you run an experiment).
Does not overwrite existing files. Writes a short README in Data/JetLag/ describing the layout.
"""
function ensure_jetlag_data_dirs()
    dirs = [
        JETLAG_DATA_ROOT,
        joinpath(JETLAG_DATA_ROOT, DIR_INITIAL_PARTICLES),
        joinpath(JETLAG_DATA_ROOT, DIR_REGULAR_SCHEDULE),
        regular_schedule_particle_hist_dir(),
        regular_schedule_particle_summaries_dir(),
        regular_schedule_cross_correlations_dir(),
        regular_schedule_shap_dir(),
        regular_schedule_particle_hist_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        regular_schedule_particle_summaries_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        regular_schedule_cross_correlations_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        regular_schedule_shap_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        jetlag_analysis_dir(),
        jetlag_analysis_particle_hist_dir(),
        joinpath(jetlag_analysis_dir(), SUBDIR_STEADY_STATE),
        jetlag_analysis_trajectories_dir(),
        jetlag_analysis_shap_dir(),
        jetlag_analysis_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        jetlag_analysis_particle_hist_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        joinpath(jetlag_analysis_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT), SUBDIR_STEADY_STATE),
        jetlag_analysis_trajectories_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        jetlag_analysis_shap_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT),
        joinpath(jetlag_analysis_dir(), SUBDIR_LIGHT_DIETS),
        joinpath(jetlag_analysis_dir(condition=REGULAR_SCHEDULE_CONDITION_HIGHLIGHT), SUBDIR_LIGHT_DIETS),
        intervention_base_dir(),
        joinpath(intervention_base_dir(), SUBDIR_PARTICLE_HISTORIES),
        intervention_adaptations_dir(),
        joinpath(intervention_base_dir(), SUBDIR_MINI, SUBDIR_PARTICLE_HISTORIES),
        joinpath(JETLAG_DATA_ROOT, "Verification"),
    ]
    for d in dirs
        mkpath(d)
    end
    readme = joinpath(JETLAG_DATA_ROOT, "README.md")
    if !isfile(readme)
        open(readme, "w") do io
            println(io, "# Jet Lag Experiment Data")
            println(io, "")
            println(io, "Generated by `ensure_jetlag_data_dirs()` in experiments/jetlag_reference.jl.")
            println(io, "")
            println(io, "| Directory | Purpose |")
            println(io, "|-----------|---------|")
            println(io, "| 1_InitialParticles | Initial particle ensembles (e.g. initial_particles1000.jld2) |")
            println(io, "| 2_RegularScheduleAnalysis | Regular schedule: particleHistories/, particleSummaries/, crossCorrelations/, SHAP/ |")
            println(io, "| 3_JetLagAnalysis | Baseline jet lag runs, steady-state results, SHAP, light diets |")
            println(io, "| 4_JetLagIntervention | Rescue policy runs: particle histories + Adaptations/ |")
            println(io, "| Verification | Verification script outputs (reports, plots) |")
            println(io, "")
            println(io, "Path and filename constants are defined in jetlag_reference.jl.")
        end
    end
    return dirs
end

# ================================================================
# SIMULATION SETTINGS
# ================================================================

# Baseline (first timezone) before shift. Revision: 70 days, so particles have settled before scoring
# (days 56–70) and before any time-zone shift (was 35 days, when many 150 lux particles were still drifting).
const BASELINE_PST_DAYS = 70
const BASELINE_PST_HOURS = Float64(BASELINE_PST_DAYS * 24)

# Post-shift target duration (days in new timezone); matches paper (50 days after travel)
const TARGET_DAYS_FULL = 50
const TARGET_DAYS_MINI = 20

# Integration and time
const DT_HOURS = 1/60
const START_DATE = DateTime(2023, 1, 1, 0, 0, 0)

# Timezone shifts (hours) for full runs
const TIMEZONE_SHIFTS_FULL = collect(1:23)
# Mini runs use a subset (e.g. [1, 6])
const TIMEZONE_SHIFTS_MINI = [1, 6]

# Particle counts
const N_PARTICLES_DEFAULT = 1000
const N_PARTICLES_MINI = 50

# ================================================================
# INITIAL PARTICLES (1_GenerateInitialParticles)
# ================================================================
# Used by initializeParticles.jl. Start date and particle count from above.
const INITIALIZATION_STATE_PARAMS_MODE = :sampled_normal  # :sampled_normal | :sampled_uniform | :fixed | etc.

# ================================================================
# ANALYSIS INPUTS (2_RegularScheduleAnalysis, 2b_SensitivityRevision)
# ================================================================
"""
    varying_inputs(input_data) -> BitVector

Which analysis inputs vary between particles. ψ is fixed at 0 for every particle (standard Process C;
`phi_sd = 0` in `initialize_particles`), so it drops out of the correlation, SHAP and RF analyses;
a ψ prior SD > 0 brings it back.
"""
varying_inputs(input_data) = BitVector([length(unique(v)) > 1 for v in input_data])

"""`perm` (a display order over the full input list) restricted to the inputs in `keep`, as indices into the kept inputs."""
restrict_order(perm, keep) = [count(keep[1:i]) for i in perm if keep[i]]

# ================================================================
# REGULAR SCHEDULE MLE SUMMARY (2_RegularScheduleAnalysis)
# ================================================================
# Used by regularSchedule_MLE_particleSummary.jl: last 14 days of the BASELINE_PST_DAYS histories.

const MLE_WINDOW_DAYS = 14
"""First day of the MLE window: the last MLE_WINDOW_DAYS of the baseline (was 20.333 with the 35-day baseline)."""
const MLE_START_DAY = Float64(BASELINE_PST_DAYS - MLE_WINDOW_DAYS)
const MLE_MIN_SLEEP_EPISODES = 13
const MLE_TIME_SHIFT = 12.0

# ================================================================
# STEADY-STATE DETECTION (shared with 3_JetLagAnalysis/detectSteadyState.jl)
# ================================================================

const STEADY_STATE_THRESHOLD = 0.25   # hours (15 min) – range/std for stability
const MIN_CONSECUTIVE_DAYS = 7        # minimum days in steady state
const EXPECTED_CBTMIN_TOLERANCE = 0.25  # hours – tolerance for expected CBTmin match

# ================================================================
# LOCATION (baseline PST: used for regular schedule and light diet)
# ================================================================

"""Default location for baseline/regular schedule (San Francisco, America/Los_Angeles)."""
const BASELINE_LOC = (lat=37.7749, lon=-122.4194, tz="America/Los_Angeles")

# ================================================================
# REGULAR SCHEDULE (2_RegularScheduleAnalysis)
# ================================================================
# Used by generateRegularScheduleParticleHistories.jl and regularSchedule_MLE_particleSummary.jl.

const REGULAR_SCHEDULE_LIGHT_DIET = "Regular_V4_061hEarlier"   # 16 h light; clock from REGULAR_SCHEDULE_LIGHT_ON_H/OFF; 10k
const REGULAR_SCHEDULE_LIGHT_DIET_150LUX = "Regular_V4_061hEarlier_150lux"  # same schedule, 150 lux

"""Light diet name for regular schedule (HighLight10000lux or LowLight150lux)."""
function regular_schedule_light_diet_for_condition(condition::String)
    return condition == REGULAR_SCHEDULE_CONDITION_LOWLIGHT ? REGULAR_SCHEDULE_LIGHT_DIET_150LUX : REGULAR_SCHEDULE_LIGHT_DIET
end

const REGULAR_SCHEDULE_SLEEP_DIET = "open"                     # "open" | "forced_sleep" | "forced_wake"
const REGULAR_SCHEDULE_SOLAR_MODEL = missing                   # or "lux_20260117", "lux_AnglePos_AdjTerms", etc.
const REGULAR_SCHEDULE_VERBOSE = true                          # verbose output from simulate_particle_population_events_only

# ================================================================
# CIRCADIAN TRANSITION (transition API)
# ================================================================
# CircadianPOMDP uses transition(state, action, params; kwargs...) as the main entry point;
# it dispatches on action type (CircadianAction vs CircadianActionExt).
#
# For JetLag we use the named variants so options are explicit:
#   • circadian_transition_sleep — precomputed light (CircadianAction): deterministic lux; eyelid fraction during sleep.
#   • circadian_transition_ext — on-the-fly light (CircadianActionExt): stochastic illuminance by default; eyelid fraction during sleep.
#   All runs use stochastic light: `regular_schedule_transition` (on-the-fly) and `jetlag_transition`
#   (precomputed light), both defined in revision_baseline.jl, with SIMULATION_SEED.
# Pass the function by name as transition_fn (no lambda needed).

"""All runs use stochastic light (Appendix A.4.1), including the jet-lag light intervention."""
const TRANSITION_DETERMINISTIC_LIGHT = false
"""Seed for every simulation; particle i draws light noise from StableRNG(SIMULATION_SEED + i)."""
const SIMULATION_SEED = 2025_02_01
const TRANSITION_SLEEP_LIGHT_FRACTION = 0.03   # matches CircadianPOMDP.EYELID_LIGHT_FRACTION (Hong et al. eyelid transmission)
const TRANSITION_DYNAMICS_MODE = :sleep_circadian  # :sleep_circadian | :circadian_only
const TRANSITION_C_PHASE_MODE = :phase_direct      # :phase_hours | :phase_direct

# ================================================================
# LIGHT CONDITIONS (names and baseline timing)
# ================================================================

const LIGHT_CONDITION_NAMES = ("LowLight150lux", "HighLight10000lux")
# Baseline PST light window (NHANES wake–aligned; same clock for array-based jet-lag diets and Regular_V4 on-the-fly)
const LIGHT_ON_BASELINE_H   = REGULAR_SCHEDULE_LIGHT_ON_H    # from CircadianPOMDP
const LIGHT_OFF_BASELINE_H  = REGULAR_SCHEDULE_LIGHT_OFF_H   # from CircadianPOMDP
const MEAN_ALIGNMENT_SHIFT_H = JETLAG_MEAN_ALIGNMENT_SHIFT_H  # from CircadianPOMDP

"""Path to jet lag particle history JLD2 for a given timezone shift (hours)."""
function particle_hist_filename_for_shift(tz_shift::Int; condition::String=REGULAR_SCHEDULE_CONDITION_DEFAULT, total_days::Int=BASELINE_PST_DAYS + TARGET_DAYS_FULL)
    dir = jetlag_analysis_particle_hist_dir(condition=condition)
    joinpath(dir, "particleHist_jetLag_$(total_days)days_$(lpad(tz_shift, 2, '0'))TimeZone_$(JETLAG_ALIGNMENT_FILENAME_SUFFIX)_mean.jld2")
end

"""Light condition specs for unified rescue pipeline (name, description, lux values)."""
function light_conditions_specs()
    return [
        (name = "LowLight150lux",
         description = "Low Light (150 lux baseline & target; manuscript default)",
         baseline_lux = 150.0,
         target_lux = 150.0,
         intervention_lux = 10000.0),
        (name = "HighLight10000lux",
         description = "High Light (10,000 lux baseline & target)",
         baseline_lux = 10000.0,
         target_lux = 10000.0,
         intervention_lux = 10000.0),
    ]
end

# ================================================================
# PLOT / OUTPUT PATHS (e.g. for plot scripts writing under 4_JetLagIntervention)
# ================================================================

"""Directory for plots/outputs written by jet lag plot scripts."""
intervention_output_dir() = intervention_base_dir()

# ================================================================
# REVISION BASELINE (regular schedule with stochastic light, entrainment rule, shared particle set)
# ================================================================
include(joinpath(@__DIR__, "revision_baseline.jl"))

# Test the circadian transition by simulating particles and plotting trajectories
# for each state variable. Two test suites:
# 1. Init methods: one run per (phase_mode, state_params_mode) with the regular
#    light diet (25 particles, 25 days).
# 2. Light diets: one run per available light diet with fixed init (random phase,
#    sampled_normal params, 25 particles, 25 days).
#
# Run from repo root: julia test/plot_transition_trajectories.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))  # project root
using Dates
using Random
using Plots
using CircadianPOMDP

include(joinpath(@__DIR__, "test_utils.jl"))

const N_PARTICLES = 25
const DURATION_DAYS = 25
const DT_HOURS = 1/60   # 1-minute steps (use 1/6 for 10-min steps if you want a faster run)
const LIGHT_DIET = "Regular_V4_061hEarlier"  # NHANES wake–aligned photoperiod; 10k lux
const SLEEP_DIET = "open"
const SOLAR_MODEL = "lux_AnglePos_AdjTerms"  # solar irradiance model for light diets that use it

# All initialization methods to test: (phase_mode, state_params_mode)
# phase_mode: :phased (uniform spacing), :random, :fixed (default phase)
# state_params_mode: :default (fixed), :sampled_uniform, :sampled_normal
const INIT_METHODS = [
    (:phased, :default),
    (:phased, :sampled_uniform),
    (:phased, :sampled_normal),
    (:random, :default),
    (:random, :sampled_uniform),
    (:random, :sampled_normal),
    (:fixed, :default),
    (:fixed, :sampled_uniform),
    (:fixed, :sampled_normal),
]

"""Sanitize a light diet name for use in filenames (alphanumeric and underscore only)."""
function sanitize_light_diet_filename(name::String)
    return replace(name, r"[^A-Za-z0-9_\-.]" => "_")
end

function run_simulation_and_plot(
    phase_mode::Symbol,
    state_params_mode::Symbol,
    n_particles::Int,
    duration_days::Int,
    start_date::DateTime,
    params;
    seed::Int=42,
    out_dir=test_output_dir(),
    light_diet::String=LIGHT_DIET,
    sleep_diet::String=SLEEP_DIET,
)
    particles = initialize_particles(
        n_particles,
        start_date;
        phase_mode=phase_mode,
        state_params_mode=state_params_mode,
        seed=seed,
    )
    stop_date = start_date + Day(duration_days)
    action = CircadianActionExt(
        light_diet,
        sleep_diet,
        SOLAR_MODEL,
        DT_HOURS,
        start_date,
        stop_date,
    )
    num_steps = get_action_num_steps(action)
    actions = fill(action, num_steps)
    particle_histories = simulate_particle_from_initial_population(
        particles,
        actions,
        params,
        circadian_transition_ext,
    )
    plt = plot_population_with_neuronal_dynamics(particle_histories; dt_hours=Float64(DT_HOURS))
    label = "phase_$(phase_mode)_params_$(state_params_mode)"
    save_path = joinpath(
        out_dir,
        "transition_trajectories_$(sanitize_light_diet_filename(light_diet))_$(label)_$(n_particles)particles_$(duration_days)days.png",
    )
    savefig(plt, save_path)
    return save_path
end

function main()
    Random.seed!(42)
    start_date = DateTime(2023, 1, 1, 0, 0, 0)
    params = model_parameters()
    out_dir = test_output_dir()

    total_runs = length(INIT_METHODS) + 1
    # Single run with default-style init (original behavior)
    println("Run 1/$total_runs: phase_mode=:random, state_params_mode=:sampled_uniform")
    path0 = run_simulation_and_plot(
        :random, :sampled_uniform, N_PARTICLES, DURATION_DAYS, start_date, params; out_dir=out_dir,
    )
    println("  Saved: $path0")

    # One run per initialization method (same light diet throughout)
    for (idx, (phase_mode, state_params_mode)) in enumerate(INIT_METHODS)
        println("Run $(idx + 1)/$total_runs: phase_mode=:$phase_mode, state_params_mode=:$state_params_mode")
        path = run_simulation_and_plot(
            phase_mode,
            state_params_mode,
            N_PARTICLES,
            DURATION_DAYS,
            start_date,
            params;
            seed=42 + idx,
            out_dir=out_dir,
        )
        println("  Saved: $path")
    end

    # One run per light diet: random phase, sampled_normal params, 25 particles, 25 days
    light_diets = get_available_light_diets()
    total_light_runs = length(light_diets)
    println("\nLight diet tests ($total_light_runs diets): phase=:random, params=:sampled_normal, $(N_PARTICLES) particles, $(DURATION_DAYS) days")
    for (idx, diet) in enumerate(light_diets)
        println("  Light diet $idx/$total_light_runs: $diet")
        path = run_simulation_and_plot(
            :random,
            :sampled_normal,
            N_PARTICLES,
            DURATION_DAYS,
            start_date,
            params;
            seed=100 + idx,
            out_dir=out_dir,
            light_diet=diet,
            sleep_diet=SLEEP_DIET,
        )
        println("    Saved: $path")
    end

    println("Done. Plots saved under $(out_dir)")
end

main()

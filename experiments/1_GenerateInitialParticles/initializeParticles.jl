# Generate initial particles
# Lara Weed
# 11 Aug 2025
#
# Paths and constants from jetlag_reference.jl

using Random
using JLD2
using Dates
using CircadianPOMDP

include(joinpath(@__DIR__, "..", "jetlag_reference.jl"))

Random.seed!(2025_01_01)

println("Initializing $(N_PARTICLES_DEFAULT) random particles... ")
N_particles = N_PARTICLES_DEFAULT
t0 = START_DATE
initial_particles = CircadianPOMDP.initialize_particles(
    N_particles, t0;
    phase_mode=:random,
    state_params_mode=INITIALIZATION_STATE_PARAMS_MODE,
)

println("Saving particles... ")
ensure_jetlag_data_dirs()  # ensure Data/JetLag/ layout exists
out_path = initial_particles_path(N_particles)
mkpath(dirname(out_path))
@save out_path initial_particles
println("Saved $(length(initial_particles)) particles to $(out_path)")
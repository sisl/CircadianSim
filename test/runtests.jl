using CircadianPOMDP
using Test

# Ensure test output directory exists
mkpath(joinpath(@__DIR__, "output"))

@testset "CircadianPOMDP" begin
    include("circadian_model_tests.jl")
    include("initialization_tests.jl")
    include("light_diet_tests.jl")
    include("transition_tests.jl")
    include("extended_transition_tests.jl")
    include("sleep_cbtmin_tests.jl")
    include("simulation_tests.jl")
    include("particle_metadata_tests.jl")
    include("particle_summaries_tests.jl")
    include("mle_summaries_tests.jl")
    include("summary_histograms_tests.jl")
    include("jetlag_integration_tests.jl")
    include("steady_state_tests.jl")
end

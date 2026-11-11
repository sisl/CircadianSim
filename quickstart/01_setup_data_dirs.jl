#!/usr/bin/env julia
# Run from repo root:  julia --project=. quickstart/01_setup_data_dirs.jl
# Creates Data/JetLag/ directory layout. Run once before the paper experiments (steps 2–4).

using CircadianPOMDP

const REPO_ROOT = dirname(@__DIR__)

println("CircadianPOMDP — setup data directories")
println("=========================================")
cd(REPO_ROOT) do
    include(joinpath("experiments", "jetlag_reference.jl"))
    dirs = ensure_jetlag_data_dirs()
    println()
    println("Created $(length(dirs)) directories under Data/JetLag/:")
    println("  1_InitialParticles")
    println("  2_RegularScheduleAnalysis/ (particleHistories, particleSummaries, crossCorrelations, SHAP)")
    println("  3_JetLagAnalysis/ (particleHistories, SteadyState, trajectories, SHAP, LightDiets)")
    println("  4_JetLagIntervention/ (particleHistories, Adaptations, Mini/particleHistories)")
    println("  Verification")
    println()
    println("Next: run the paper pipeline in this order (see \"Reproducing the paper\" in README.md)")
    println("  1. julia --project=. experiments/1_GenerateInitialParticles/initializeParticles.jl")
    println("  2. zsh experiments/2b_SensitivityRevision/run_exp2_pipeline.sh")
    println("  3. zsh experiments/2b_SensitivityRevision/run_exp3_pipeline.sh trajectories")
    println("  4. julia --project=. -t auto experiments/4_JetLagPolicy/rescuingJetLag_unified.jl mini   # or without 'mini' for the full run")
    println()
end

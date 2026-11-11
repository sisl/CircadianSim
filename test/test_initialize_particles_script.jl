#!/usr/bin/env julia
# =============================================================================
# Standalone script to verify the unified initialize_particles() API.
# Run from repo root: julia --project=. test/test_initialize_particles_script.jl
#
# Loads only the source files needed for initialization (circadian_model.jl,
# initialization_functions.jl) so it runs even when the full package does not
# precompile (e.g. CircadianStateExt issue). No need to load CircadianPOMDP.
#
# Possible additions (API or script):
#   - seed / rng: reproducible sampling per call without touching global RNG.
#   - sleep override: e.g. sleep=false for :sampled when you want varied params but all awake.
#   - tau_x for one-per-location: optional kwarg to match legacy initialize_particles_one_per_location(t, tau_x, locations).
#   - N=0: return CircadianState[] and document.
#   - Error-path tests: wrong signature for one_per_location, invalid location_mode.
#   - Edge cases: N=1; optional --quiet for CI.
# =============================================================================

using Random
using Dates
using Printf
using Distributions

# Load minimal src needed for initialize_particles (avoids full package load)
const _script_dir = @__DIR__
const _project_root = abspath(joinpath(_script_dir, ".."))
const _src_dir = joinpath(_project_root, "src")
include(joinpath(_src_dir, "circadian_model.jl"))
include(joinpath(_src_dir, "initialization_functions.jl"))

const t0 = DateTime(2023, 1, 1, 0, 0, 0)

function run_section(fn, name::String)
    print("  ")
    printstyled(name; color=:cyan)
    println(" ...")
    try
        fn()
        print("    ")
        printstyled("OK\n"; color=:green)
        return true
    catch e
        print("    ")
        printstyled("FAIL: $e\n"; color=:red)
        rethrow(e)
    end
end

function run_section_expect_error(fn, name::String)
    print("  ")
    printstyled(name; color=:cyan)
    println(" ...")
    try
        fn()
        print("    ")
        printstyled("FAIL: expected an error\n"; color=:red)
        rethrow(AssertionError("expected an error"))
    catch e
        if e isa AssertionError && occursin("expected an error", string(e))
            rethrow(e)
        end
        print("    ")
        printstyled("OK (error as expected)\n"; color=:green)
        return true
    end
end

function main()
    Random.seed!(42)
    println()
    printstyled("=== initialize_particles() verification script ===\n\n"; color=:bold)

    # -------------------------------------------------------------------------
    # 1. Single location, default options (random phase, default params)
    # -------------------------------------------------------------------------
    run_section("1. N particles, single location (default: random phase, default params)") do
        p = initialize_particles(100, t0)
        @assert length(p) == 100
        @assert all(x -> x isa CircadianState, p)
        @assert all(x -> isapprox(x.x^2 + x.xc^2, 1.0, atol=1e-10), p)
        @assert all(x -> x.latitude ≈ 37.4277 && x.longitude ≈ -122.1701, p)
        @assert all(x -> x.tau_x ≈ 24.2 && x.H ≈ 11.0, p)
        @assert all(x -> x.t == t0, p)
    end

    # -------------------------------------------------------------------------
    # 2. Phase modes: phased
    # -------------------------------------------------------------------------
    run_section("2. phase_mode=:phased (uniform spacing on unit circle)") do
        N = 12
        p = initialize_particles(N, t0; phase_mode=:phased, state_params_mode=:default)
        @assert length(p) == N
        for i in 1:N
            φ = 2π * (i - 1) / N
            @assert isapprox(p[i].x, cos(φ), atol=1e-10)
            @assert isapprox(p[i].xc, sin(φ), atol=1e-10)
        end
    end

    # -------------------------------------------------------------------------
    # 3. Phase mode: fixed
    # -------------------------------------------------------------------------
    run_section("3. phase_mode=:fixed (all at default phase)") do
        p = initialize_particles(20, t0; phase_mode=:fixed)
        @assert length(p) == 20
        @assert all(x -> x.x ≈ -1.22 && x.xc ≈ -0.17, p)
    end

    # -------------------------------------------------------------------------
    # 4. State params: sampled with custom ranges
    # -------------------------------------------------------------------------
    run_section("4. state_params_mode=:sampled with custom tau_range") do
        p = initialize_particles(50, t0; state_params_mode=:sampled, tau_range=(24.0, 24.1), seed=7)
        @assert length(p) == 50
        @assert all(x -> 24.0 <= x.tau_x <= 24.1, p)
        @assert all(x -> 10.0 <= x.H <= 16.0, p)
        @assert all(x -> 0.8 <= x.Vm <= 1.2 && -10.5 <= x.Vv <= -9.5, p)
        @assert 0 < count(s -> s.sleep, p) < 50  # default Bernoulli(1/3)
        @assert all(s -> !s.sleep, initialize_particles(5, t0; state_params_mode=:sampled, initial_sleep_prob=0.0, seed=1))
        @assert all(s -> s.sleep, initialize_particles(5, t0; state_params_mode=:sampled, initial_sleep_prob=1.0, seed=1))
        n_s = count(s -> s.sleep, initialize_particles(400, t0; state_params_mode=:sampled, seed=42))
        @assert 80 <= n_s <= 200  # Binomial(400,1/3); wide tolerance
    end

    # -------------------------------------------------------------------------
    # 5. Custom single location (latitude, longitude, timezone)
    # -------------------------------------------------------------------------
    run_section("5. Custom single location (lat/lon/tz)") do
        p = initialize_particles(10, t0; latitude=40.7, longitude=-74.0, timezone="America/New_York")
        @assert length(p) == 10
        @assert all(x -> x.latitude ≈ 40.7 && x.longitude ≈ -74.0, p)
        @assert all(x -> x.timezone == "America/New_York", p)
    end

    # -------------------------------------------------------------------------
    # 6. One particle per location (signature: t, locations)
    # -------------------------------------------------------------------------
    run_section("6. One per location: initialize_particles(t, locations; location_mode=:one_per_location)") do
        locs = [
Location("NYC", 40.7, -74.0, "America/New_York", 1.0),
Location("LA", 34.0, -118.2, "America/Los_Angeles", 1.0),
Location("Chicago", 41.9, -87.6, "America/Chicago", 1.0),
        ]
        p = initialize_particles(t0, locs; location_mode=:one_per_location)
        @assert length(p) == 3
        @assert p[1].latitude ≈ 40.7 && p[1].timezone == "America/New_York"
        @assert p[2].latitude ≈ 34.0 && p[2].timezone == "America/Los_Angeles"
        @assert p[3].latitude ≈ 41.9 && p[3].timezone == "America/Chicago"
    end

    # -------------------------------------------------------------------------
    # 7. Multiple locations, even distribution
    # -------------------------------------------------------------------------
    run_section("7. N particles, locations, location_mode=:even") do
        locs = [
Location("A", 40.0, -100.0, "America/Chicago", 1.0),
Location("B", 35.0, -95.0, "America/Chicago", 1.0),
        ]
        p = initialize_particles(200, t0; locations=locs, location_mode=:even)
        @assert length(p) == 200
        lats = [s.latitude for s in p]
        @assert all(lat -> lat ≈ 40.0 || lat ≈ 35.0, lats)
        # With seed, we can expect both locations to appear
        @assert 1 <= count(==(40.0), lats) <= 199
        @assert 1 <= count(==(35.0), lats) <= 199
    end

    # -------------------------------------------------------------------------
    # 8. Multiple locations, weighted distribution
    # -------------------------------------------------------------------------
    run_section("8. N particles, locations, location_mode=:weighted") do
        locs = [
Location("rare", 40.0, -100.0, "America/Chicago", 1.0),
Location("common", 35.0, -95.0, "America/Chicago", 9.0),
        ]
        p = initialize_particles(500, t0; locations=locs, location_mode=:weighted)
        @assert length(p) == 500
        n_rare = count(s -> s.latitude ≈ 40.0, p)
        n_common = count(s -> s.latitude ≈ 35.0, p)
        # Weighted 1:9 => expect ~10% rare, ~90% common (allow wide margin)
        @assert 20 <= n_rare <= 150
        @assert 350 <= n_common <= 480
        @printf("    (rare: %d, common: %d)\n", n_rare, n_common)
    end

    # -------------------------------------------------------------------------
    # 9. Combined: phased + sampled + weighted locations
    # -------------------------------------------------------------------------
    run_section("9. Combined: phased + sampled params + weighted locations") do
        locs = [
Location("X", 45.0, -93.0, "America/Chicago", 1.0),
Location("Y", 42.0, -88.0, "America/Chicago", 1.0),
        ]
        p = initialize_particles(20, t0;
            phase_mode=:phased,
            state_params_mode=:sampled,
            tau_range=(24.0, 24.2),
            locations=locs,
            location_mode=:weighted,
        )
        @assert length(p) == 20
        for i in 1:20
            φ = 2π * (i - 1) / 20
            @assert isapprox(p[i].x, cos(φ), atol=1e-10)
            @assert isapprox(p[i].xc, sin(φ), atol=1e-10)
        end
        @assert all(x -> 24.0 <= x.tau_x <= 24.2, p)
        @assert all(x -> x.latitude ≈ 45.0 || x.latitude ≈ 42.0, p)
    end

    # -------------------------------------------------------------------------
    # 10. Reproducibility with seed
    # -------------------------------------------------------------------------
    run_section("10. Reproducibility with seed") do
        p1 = initialize_particles(30, t0; state_params_mode=:sampled, seed=12345)
        p2 = initialize_particles(30, t0; state_params_mode=:sampled, seed=12345)
        @assert length(p1) == length(p2) == 30
        for i in 1:30
            @assert p1[i].x ≈ p2[i].x && p1[i].xc ≈ p2[i].xc
            @assert p1[i].tau_x ≈ p2[i].tau_x && p1[i].H ≈ p2[i].H
        end
        # Different seed -> different particles (at least one differing)
        p3 = initialize_particles(30, t0; state_params_mode=:sampled, seed=99999)
        @assert any(i -> p1[i].x != p3[i].x || p1[i].tau_x != p3[i].tau_x, 1:30)
    end

    # -------------------------------------------------------------------------
    # 11. Phased + default params (structure check)
    # -------------------------------------------------------------------------
    run_section("11. Phased + default params") do
        N = 8
        p = initialize_particles(N, t0; phase_mode=:phased, state_params_mode=:default)
        for i in 1:N
            φ = 2π * (i - 1) / N
            @assert isapprox(p[i].x, cos(φ), atol=1e-10)
            @assert isapprox(p[i].xc, sin(φ), atol=1e-10)
            @assert p[i].tau_x ≈ 24.2 && p[i].H ≈ 11.0
        end
    end

    # -------------------------------------------------------------------------
    # 12. Edge cases: N=0 and empty locations
    # -------------------------------------------------------------------------
    run_section("12. N=0 returns empty vector") do
        p = initialize_particles(0, t0)
        @assert p isa Vector
        @assert length(p) == 0
    end

    run_section("13. One per location with empty locations returns empty vector") do
        p = initialize_particles(t0, Location[]; location_mode=:one_per_location)
        @assert length(p) == 0
    end

    # -------------------------------------------------------------------------
    # 14–17. Validation: invalid inputs must throw clear errors
    # -------------------------------------------------------------------------
    run_section_expect_error("14. Reversed tau_range with :sampled throws") do
        initialize_particles(5, t0; state_params_mode=:sampled, tau_range=(24.4, 24.0))
    end

    run_section_expect_error("15. Invalid latitude throws") do
        initialize_particles(5, t0; latitude=100.0, longitude=0.0)
    end

    run_section_expect_error("16. Empty timezone throws") do
        initialize_particles(5, t0; timezone="")
    end

    run_section_expect_error("17. Invalid phase_mode throws") do
        initialize_particles(5, t0; phase_mode=:phase)
    end

    println()
    printstyled("All checks passed. New initialize_particles() behavior is as expected.\n"; color=:bold)
    println()
    return nothing
end

main()

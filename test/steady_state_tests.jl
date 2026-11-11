@testset "Steady-state detection (analysis/steady_state.jl)" begin

    # ========== unwrap24 ==========
    @testset "unwrap24" begin
        # No wrapping needed
        @test unwrap24([1.0, 2.0, 3.0]) ≈ [1.0, 2.0, 3.0]

        # Forward jump (e.g. 23 → 1 means clock wraps forward)
        u = unwrap24([22.0, 23.0, 0.5, 1.5])
        @test u[3] ≈ 24.5
        @test u[4] ≈ 25.5

        # Backward jump (e.g. 1 → 23 means clock wraps backward)
        u2 = unwrap24([1.0, 2.0, 23.0])
        @test u2[3] ≈ -1.0

        # Single element
        @test unwrap24([5.0]) == [5.0]

        # Empty
        @test unwrap24(Float64[]) == Float64[]
    end

    # ========== wrap24 ==========
    @testset "wrap24" begin
        @test wrap24(25.0) ≈ 1.0
        @test wrap24(-1.0) ≈ 23.0
        @test wrap24(0.0) ≈ 0.0
        @test wrap24(24.0) ≈ 0.0
        @test wrap24(48.5) ≈ 0.5
        @test wrap24(12.0) ≈ 12.0
    end

    # ========== is_steady_state_window ==========
    @testset "is_steady_state_window" begin
        # Stable: small range
        @test is_steady_state_window([5.0, 5.05, 5.02, 5.01, 5.03, 5.02, 5.01])

        # Unstable: large range
        @test !is_steady_state_window([1.0, 5.0, 10.0, 15.0])

        # Too few values
        @test !is_steady_state_window([5.0])
        @test !is_steady_state_window(Float64[])

        # Just barely within threshold
        @test is_steady_state_window([5.0, 5.24]; threshold=0.25)

        # Just barely outside threshold (also check std)
        vals = [5.0, 5.3]  # range=0.3, std≈0.21 > 0.125
        @test !is_steady_state_window(vals; threshold=0.25)

        # Wrapping: values near midnight that are actually close
        @test is_steady_state_window([23.9, 0.1, 0.0, 23.95])
    end

    # ========== find_baseline_steady_state_cbtmin ==========
    @testset "find_baseline_steady_state_cbtmin" begin
        # Stable tail
        traj = vcat(collect(1.0:0.5:10.0), fill(5.0, 10))
        result = find_baseline_steady_state_cbtmin(traj; min_consecutive_days=7)
        @test result !== nothing
        @test result ≈ 5.0

        # All unstable
        @test find_baseline_steady_state_cbtmin(collect(1.0:1.0:20.0); min_consecutive_days=7) === nothing

        # Too short
        @test find_baseline_steady_state_cbtmin([5.0, 5.0, 5.0]; min_consecutive_days=7) === nothing

        # Wrapping near midnight
        traj_wrap = vcat(fill(12.0, 5), fill(23.95, 10))
        result_wrap = find_baseline_steady_state_cbtmin(traj_wrap; min_consecutive_days=7)
        @test result_wrap !== nothing
        @test abs(result_wrap - 23.95) < 0.1
    end

    # ========== find_first_steady_state_day ==========
    @testset "find_first_steady_state_day" begin
        # Baseline at 5.0, shift by +3 → expect 8.0
        baseline_count = 10
        baseline = fill(5.0, baseline_count)
        post = vcat(collect(range(5.0, 8.0, length=5)), fill(8.0, 10))
        trajectory = vcat(baseline, post)

        days, mean_val, dir = find_first_steady_state_day(
            trajectory, 5.0, 3, baseline_count;
            min_consecutive_days=7, expected_tolerance=0.5)
        @test days !== nothing
        @test mean_val !== nothing
        @test abs(mean_val - 8.0) < 0.5
        @test dir isa Symbol

        # No steady state reached (all different)
        noisy = vcat(fill(5.0, 10), collect(1.0:0.5:10.0))
        d2, m2, dir2 = find_first_steady_state_day(noisy, 5.0, 3, 10; min_consecutive_days=7)
        @test d2 === nothing

        # Too short trajectory
        d3, m3, dir3 = find_first_steady_state_day([5.0, 5.0], 5.0, 3, 1; min_consecutive_days=7)
        @test d3 === nothing
    end

    # ========== compute_adaptation_time ==========
    @testset "compute_adaptation_time" begin
        baseline_count = 10
        baseline = fill(5.0, baseline_count)
        post = vcat(collect(range(5.0, 8.0, length=5)), fill(8.0, 10))
        trajectory = vcat(baseline, post)

        result = compute_adaptation_time(trajectory, 5.0, baseline_count, 3;
                                         min_consecutive_days=7, expected_tolerance=0.5)
        @test result !== nothing
        @test haskey(result, :time)
        @test haskey(result, :direction)
        @test haskey(result, :baseline_cbtmin)
        @test haskey(result, :steady_cbtmin)
        @test result.baseline_cbtmin ≈ 5.0

        # No steady state
        @test compute_adaptation_time([5.0, 6.0, 7.0], 5.0, 2, 3; min_consecutive_days=7) === nothing
    end

    # ========== normalize_time! / normalize_time ==========
    @testset "normalize_time" begin
        v = [0.5, 12.0, 14.0, 22.0]
        result = normalize_time!(copy(v))
        @test result ≈ [0.5, -12.0, -10.0, -2.0]

        # Mutating
        v2 = [13.0, 23.5]
        normalize_time!(v2)
        @test v2 ≈ [-11.0, -0.5]

        # Non-mutating
        v3 = [6.0, 18.0]
        out = normalize_time(v3)
        @test out ≈ [6.0, -6.0]
        @test v3 == [6.0, 18.0]  # original unchanged
    end
end

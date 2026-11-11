using Random
using Dates
using CircadianPOMDP

@testset "initialization" begin
    Random.seed!(42)
    t0 = DateTime(2023, 1, 1, 0, 0, 0)

    # initialize_particles: phased (replaces initialize_particles_phased)
    p_phased = CircadianPOMDP.initialize_particles(10, t0; phase_mode=:phased, state_params_mode=:default)
    @test length(p_phased) == 10
    @test all(p -> isapprox(p.x^2 + p.xc^2, 1.0, atol=1e-10), p_phased)
    @test all(p -> p.tau_x ≈ 24.2, p_phased)
    @test all(p -> p.K ≈ 0.55, p_phased)

    # initialize_particles: random + sampled (replaces initialize_particles_random)
    p_rand = CircadianPOMDP.initialize_particles(50, t0; state_params_mode=:sampled)
    @test length(p_rand) == 50
    @test all(p -> isapprox(p.x^2 + p.xc^2, 1.0, atol=1e-10), p_rand)
    @test all(p -> 10.0 <= p.H <= 16.0, p_rand)
    @test all(p -> 0.8 <= p.Vm <= 1.2, p_rand)
    @test all(p -> p.tau_x > 20 && p.tau_x < 28, p_rand)
    @test all(p -> p.K > 0 && p.chi > 0 && p.mu_h > 0, p_rand)

    # initialize_particles: custom tau_range (replaces initialize_particles_custom)
    p_custom = CircadianPOMDP.initialize_particles(20, t0; state_params_mode=:sampled, tau_range=(24.0, 24.1))
    @test length(p_custom) == 20
    @test all(p -> 24.0 <= p.tau_x <= 24.1, p_custom)

    # initialize_particles: single location
    p_single = CircadianPOMDP.initialize_particles(10, t0; latitude=37.77, longitude=-122.42, timezone="America/Los_Angeles", state_params_mode=:sampled)
    @test length(p_single) == 10
    @test all(p -> p.latitude ≈ 37.77 && p.timezone == "America/Los_Angeles", p_single)

    # initialize_particles — default API
    p_u = CircadianPOMDP.initialize_particles(50, t0)
    @test length(p_u) == 50
    @test all(p -> p isa CircadianPOMDP.CircadianState, p_u)
    @test all(p -> isapprox(p.x^2 + p.xc^2, 1.0, atol=1e-10), p_u)

    # Edge case: N=1
    p_one = CircadianPOMDP.initialize_particles(1, t0)
    @test length(p_one) == 1
    @test p_one[1] isa CircadianPOMDP.CircadianState
    @test isapprox(p_one[1].x^2 + p_one[1].xc^2, 1.0, atol=1e-10)

    # Edge case: N=0
    p_zero = CircadianPOMDP.initialize_particles(0, t0)
    @test length(p_zero) == 0

    # Error path: invalid phase_mode
    @test_throws Exception CircadianPOMDP.initialize_particles(5, t0; phase_mode=:nonexistent_mode)

    # Initial sleep: default Bernoulli(1/3); explicit p=0 / p=1
    n_asleep = count(s -> s.sleep, CircadianPOMDP.initialize_particles(300, t0; state_params_mode=:sampled, seed=11))
    @test 60 <= n_asleep <= 160  # Binomial(300, 1/3), wide band
    @test all(s -> !s.sleep, CircadianPOMDP.initialize_particles(5, t0; state_params_mode=:sampled, initial_sleep_prob=0.0, seed=1))
    @test all(s -> s.sleep, CircadianPOMDP.initialize_particles(5, t0; state_params_mode=:sampled, initial_sleep_prob=1.0, seed=1))
end

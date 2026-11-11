using Distributions
using Printf
using Dates
using Random

# ================================================================
# Particle Initialization (JetLag: single location only)
# ================================================================

"""
Default P(start asleep) for `initialize_particles`: each particle draws
`sleep ~ Bernoulli(DEFAULT_INITIAL_SLEEP_PROB_ASLEEP)` (⅓ asleep, ⅔ awake).
Override with `initial_sleep_prob` (e.g. `0.0` / `1.0` for deterministic awake / asleep).
"""
const DEFAULT_INITIAL_SLEEP_PROB_ASLEEP = 1 / 3

# =================================================================
# Validation helpers
# =================================================================

function _check_range(r, name::String)
    length(r) == 2 || error("$name must be a (lo, hi) tuple of length 2, got length $(length(r))")
    lo, hi = r[1], r[2]
    isfinite(lo) && isfinite(hi) || error("$name must have finite bounds, got ($lo, $hi)")
    lo <= hi || error("$name must have lo <= hi, got ($lo, $hi)")
end

function _check_lat_lon(lat::Float64, lon::Float64)
    -90 <= lat <= 90 || error("latitude must be in [-90, 90], got $lat")
    -180 <= lon <= 180 || error("longitude must be in [-180, 180], got $lon")
end

function _check_timezone(tz::String)
    isempty(tz) && error("timezone must be non-empty")
end

# =================================================================
# initialize_particles (single location for all particles)
# =================================================================

"""
    initialize_particles(N::Int, t::DateTime; state_params_mode=:default, ...) -> Vector{CircadianState}

Create N particles at time t with a single location (latitude, longitude, timezone) for all particles.

# Phase (circadian state x, xc)
- `phase_mode::Symbol = :random` — `:phased` (uniform spacing), `:random`, or `:fixed`
- For `:random`, `φ = 2π * rand()` maps `[0,1)` to `[0, 2π)` (no duplicate mass at `0` and `2π`).
  With `x = cos(φ)`, `xc = sin(φ)`, the **joint** distribution is uniform on the unit circle, but **marginal**
  histograms of `x` or `xc` are U-shaped (arcsine-type density near `±1`), not flat—that is expected, not a
  sampling bug. To check phase uniformity, use `atan(xc, x)` or `φ` directly.

# State parameters
- `state_params_mode::Symbol = :default` — `:default`, `:sampled_uniform`, `:sampled_normal`, or `:sampled` (alias for uniform)
- For `:sampled_normal` (recommended for JetLag) optional: `tau_mean`, `tau_sd`, `K_mean`, etc.

# Location (single for all particles)
- `latitude::Float64 = 37.4277`, `longitude::Float64 = -122.1701`, `timezone::String = "America/Los_Angeles"`

# Initial sleep
- `initial_sleep_prob::Float64 = DEFAULT_INITIAL_SLEEP_PROB_ASLEEP` — each particle draws `sleep ~ Bernoulli(p)` independently (`p` = P start asleep).

# Reproducibility
- `seed::Union{Int,Nothing} = nothing`
"""
function initialize_particles(
    N::Int,
    t::DateTime;
    phase_mode::Symbol = :random,
    state_params_mode::Symbol = :default,
    initial_sleep_prob::Float64 = DEFAULT_INITIAL_SLEEP_PROB_ASLEEP,
    tau_range = (24.0, 24.4),
    K_range = (0.4, 0.7),
    chi_range = (40.0, 50.0),
    mu_h_range = (3.5, 4.5),
    phi_range = (-0.5, 0.5),
    H_range = (10.0, 16.0),
    tau_mean::Float64 = 24.2,
    tau_sd::Float64 = 0.2,
    K_mean::Float64 = 0.55,
    K_sd::Float64 = 0.05,
    chi_mean::Float64 = 45.0,
    chi_sd::Float64 = 5.0,
    mu_h_mean::Float64 = 4.2,
    mu_h_sd::Float64 = 0.5,
    phi_mean::Float64 = 0.0,
    phi_sd::Float64 = 0.0,   # ψ fixed at 0 (standard Process C); > 0 adds a between-person phase lag
    latitude::Float64 = 37.4277,
    longitude::Float64 = -122.1701,
    timezone::String = "America/Los_Angeles",
    seed::Union{Int, Nothing} = nothing,
)
    if N < 0
        error("N must be non-negative, got N = $N")
    end

    if !(phase_mode in (:phased, :random, :fixed))
        error("phase_mode must be :phased, :random, or :fixed, got $phase_mode")
    end
    if !(state_params_mode in (:default, :sampled_uniform, :sampled_normal, :sampled))
        error("state_params_mode must be :default, :sampled_uniform, :sampled_normal, or :sampled, got $state_params_mode")
    end
    resolved_mode = state_params_mode == :sampled ? :sampled_uniform : state_params_mode

    if resolved_mode in (:sampled_uniform, :sampled_normal)
        _check_range(tau_range, "tau_range")
        _check_range(K_range, "K_range")
        _check_range(chi_range, "chi_range")
        _check_range(mu_h_range, "mu_h_range")
        _check_range(phi_range, "phi_range")
        _check_range(H_range, "H_range")
    end

    _check_lat_lon(latitude, longitude)
    _check_timezone(timezone)
    (0.0 <= initial_sleep_prob <= 1.0) ||
        error("initial_sleep_prob must be in [0, 1], got $initial_sleep_prob")

    rng = seed !== nothing ? Random.MersenneTwister(seed) : Random.GLOBAL_RNG
    particles = CircadianState[]

    for i in 1:N
        if phase_mode == :phased
            φ = 2π * (i - 1) / N
            x = cos(φ)
            xc = sin(φ)
        elseif phase_mode == :random
            φ = 2π * rand(rng)
            x = cos(φ)
            xc = sin(φ)
        else
            x = -1.22
            xc = -0.17
        end

        if resolved_mode == :default
            n, H, Vm, Vv = 0.5, 11.0, 1.09, -10.2
            H_plus, H_minus = 15.0, 15.0
            tau_x, K, chi, mu_h, phi_val = 24.2, 0.55, 45.0, 4.2, 0.0
        elseif resolved_mode == :sampled_uniform
            n = 0.5
            H = rand(rng, Uniform(H_range...))
            Vm = rand(rng, Uniform(0.8, 1.2))
            Vv = rand(rng, Uniform(-10.5, -9.5))
            H_plus, H_minus = 15.0, 15.0
            tau_x = rand(rng, Uniform(tau_range...))
            K = rand(rng, Uniform(K_range...))
            chi = rand(rng, Uniform(chi_range...))
            mu_h = rand(rng, Uniform(mu_h_range...))
            phi_val = rand(rng, Uniform(phi_range...))
        else
            n = 0.5
            H = rand(rng, Uniform(H_range...))
            Vm = rand(rng, Uniform(0.8, 1.2))
            Vv = rand(rng, Uniform(-10.5, -9.5))
            H_plus, H_minus = 15.0, 15.0
            tau_x = rand(rng, Normal(tau_mean, tau_sd))
            K = rand(rng, Normal(K_mean, K_sd))
            chi = rand(rng, Normal(chi_mean, chi_sd))
            mu_h = rand(rng, Normal(mu_h_mean, mu_h_sd))
            phi_val = rand(rng, Normal(phi_mean, phi_sd))
        end
        sleep = rand(rng, Bernoulli(initial_sleep_prob))
        I_eye = 0.0

        push!(particles, CircadianState(
            n, x, xc, H, Vm, Vv, sleep, H_plus, H_minus,
            tau_x, K, chi, mu_h, phi_val, I_eye,
            latitude, longitude, timezone, t
        ))
    end
    return particles
end

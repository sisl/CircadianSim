# CircadianPOMDP Test Suite Documentation

This document provides a detailed description of all tests in the CircadianPOMDP package. The test suite covers all major exported functions in `src/`, including unit tests and visual verification where applicable.

---

## Running the Tests

```bash
# From project root
julia --project -e 'using Pkg; Pkg.test()'

# Or include runtests directly
julia --project -e 'include("test/runtests.jl")'
```

Output directory for plots and temporary files: `test/output/`

---

## Test Structure Overview

| Test File | Testset Name | Description |
|-----------|--------------|-------------|
| `circadian_model_tests.jl` | circadian_model | Model parameters, space types, default constructors |
| `initialization_tests.jl` | initialization | Particle initialization, conversion, locations |
| `light_diet_tests.jl` | light_diet | Light diet generation, solar models, timezone utilities |
| `transition_tests.jl` | circadian_transition | Basic circadian and sleep transition dynamics |
| `extended_transition_tests.jl` | circadian_transition_ext | Extended state/action types, action windows |
| `sleep_cbtmin_tests.jl` | sleep_CBTmin_detection | CBTmin and sleep period detection |
| `simulation_tests.jl` | simulation | Particle population simulation (full and events-only) |
| `particle_metadata_tests.jl` | particle_metadata | Metrics, metadata, population statistics |
| `particle_summaries_tests.jl` | particle_summaries | Von Mises estimation, legacy particle summaries |
| `mle_summaries_tests.jl` | particle_summaries_mle | MLE population summaries, analysis, save/load |
| `summary_histograms_tests.jl` | summary_histograms | MLE summary histogram plotting |
| `jetlag_integration_tests.jl` | jetlag_integration | End-to-end jet lag pipeline (light diets, steady state) |
| `steady_state_tests.jl` | steady_state | Steady-state detection, time normalization utilities |
| `threading_tests.jl` | threading_utils | Threading status and optimal thread count (optional, not in default run) |
| `plot_tests.jl` | plot_visual_verification | Visual plots (optional, not in default run) |

---

## Detailed Test Descriptions

### 1. circadian_model_tests.jl

Tests the core model structure and default constructors.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| model_parameters | `model_parameters()` | Returns dict with `params_C`, `params_H`, `params_X`; tau_x≈24.2, K≈0.55, chi≈45, mu_h≈4.2, phi≈0 |
| get_space_types | `get_space_types()` | Returns tuple `(CircadianState, CircadianAction, CircadianObservation)` |
| default_circadian_state | `default_circadian_state()` | Returns CircadianState with x≈-1.22, xc≈-0.17, H≈11.0, tau_x≈24.2 |
| default_circadian_action | `default_circadian_action(I, dt)` | Returns CircadianAction with correct I_act and dt |
| default_circadian_observation | `default_circadian_observation(I)` | Returns CircadianObservation |
| get_all_parameters_flat | `get_all_parameters_flat()` | Returns Dict with tau_x, chi, and >10 keys |
| print_parameter_summary | `print_parameter_summary()` | Runs without error, returns nothing |

---

### 2. initialization_tests.jl

Tests particle initialization strategies and conversion to extended state.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| initialize_particles (phased) | `initialize_particles(N, t0; phase_mode=:phased, state_params_mode=:default)` | N particles, unit circle (x²+xc²≈1), tau_x≈24.2, K≈0.55 |
| initialize_particles (random+sampled) | `initialize_particles(N, t0; state_params_mode=:sampled)` | N particles, unit circle, H∈[10,16], Vm∈[0.8,1.2], tau_x∈(20,28), K/chi/mu_h>0 |
| initialize_particles (custom tau) | `initialize_particles(N, t0; state_params_mode=:sampled, tau_range=...)` | tau_x within specified range |
| initialize_particles (location) | `initialize_particles(N, t0; latitude=..., longitude=..., timezone=...)` | N particles with same location |
| initialize_particles (default) | `initialize_particles(N, t0)` | All CircadianState, unit circle |
| Edge: N=1 | `initialize_particles(1, t0)` | Single valid particle |
| Edge: N=0 | `initialize_particles(0, t0)` | Empty vector |
| Error: invalid phase_mode | `initialize_particles(N, t0; phase_mode=:nonexistent)` | Throws exception |

---

### 3. light_diet_tests.jl

Tests light diet generation, solar models, and timezone utilities.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| generate_light_diet_basic | `generate_light_diet_basic(...)` | Length = days×24×60, all CircadianAction, dt≈1/60, correct lux schedule (0→10k at 7h, 10k until 23h, then 0) |
| generate_shifted_dark_diet | `generate_shifted_dark_diet(...)` | Length = 24×60 |
| generate_light_period_diet | `generate_light_period_diet(...)` | Length = 24×60 |
| generate_intervention_light_diet | `generate_intervention_light_diet(...)` | Length = 24×60 |
| get_intervention_policies | `get_intervention_policies()` | 300 policies, valid start (0–23) and duration |
| generate_light_diets_for_condition | `generate_light_diets_for_condition(condition)` | LD_PST length, targets length (23), target day length |
| build_baseline_diet_full | `build_baseline_diet_full(...)` | Correct concatenated length |
| build_intervention_diet_full | `build_intervention_diet_full(...)` | Correct concatenated length |
| get_available_light_diets | `get_available_light_diets()` | Returns AbstractVector, includes "Regular_V4_061hEarlier" |
| get_light_intensity | `get_light_intensity("Regular_V4_061hEarlier", t, lat, solar_model)` | Returns Tuple (I_act, sleep_forced), first element ≥ 0 |
| get_timezone_center_lon | `get_timezone_center_lon("America/Los_Angeles")` | Float64 in [-180, 180] |
| infer_timezone_center_from_longitude | `infer_timezone_center_from_longitude(lon)` | Returns Float64 |
| JETLAG_MEAN_ALIGNMENT_SHIFT_H | Constant | Same as `REGULAR_SCHEDULE_NHANES_WAKE_ALIGNMENT_SHIFT_H` (1.341 h) |
| get_light_and_sleep | `get_light_and_sleep(t, lat, lon, diet, solar_model)` | I ≥ 0, sleep is Bool or Missing |
| compute_light_from_action | `compute_light_from_action(t, lat, lon, diet, solar_model)` | I ≥ 0 |

---

### 4. transition_tests.jl

Tests basic circadian transition dynamics.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| transition (dark) | `transition(s, a, params)` | Returns CircadianState, state evolves or stays, n∈[0,1], finite x/xc |
| transition (light) | With I_act=1000 | Returns CircadianState |
| circadian_transition_sleep | `circadian_transition_sleep(s, a, params)` | Returns CircadianState |
| Short trajectory | `simulate_particle_from_initial_population` | 1 particle, length = actions+1, first state preserved |

---

### 5. extended_transition_tests.jl

Tests extended state/action types and action window logic.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| is_within_action_window | `is_within_action_window(s, action)` / `(t, action)` | true for t0, false for t0+2days |
| get_action_duration_hours | `get_action_duration_hours(action)` | ≈ 24.0 |
| get_action_num_steps | `get_action_num_steps(action)` | 24×60 |
| create_action_sequence | `create_action_sequence(actions)` | Sorts by startdate |
| get_current_action | `get_current_action(t, actions)` | Returns action for t0+6h, nothing for t0+2days |
| circadian_transition_ext | `circadian_transition_ext(s_ext, action, params)` | Returns CircadianState, t advances, finite x/xc |

---

### 6. sleep_cbtmin_tests.jl

Tests CBTmin (core body temperature minimum) and sleep period detection.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| get_CBTmin | `get_CBTmin(history; dt)` | Returns Vector{Int}, indices in valid range |
| get_sleep_periods | `get_sleep_periods(history; dt)` | Returns Vector of SleepPeriod, each with onset_idx, offset_idx, duration |
| Consistency | CBTmin and sleep from same history | idx[1] ≥ 1 when non-empty |

---

### 7. simulation_tests.jl

Tests particle population simulation (full histories and events-only).

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| simulate_particle_from_initial_population | Given particles + LD + params | N histories, each length = LD+1, first state preserved, all CircadianState |
| simulate_particle_population | Generates own particles | Same structure, N particles |
| simulate_particle_population_events_only | Events-based, no full history | Returns (cbtmin_ev, onset_ev, offset_ev); each a Vector of events with particle_id ∈ 1..N |

---

### 8. particle_metadata_tests.jl

Tests metrics calculation, event-based metrics, and population metadata.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| calculate_sleep_metrics | `calculate_sleep_metrics(history; dt)` | Returns SleepMetrics, sleep_periods is Vector |
| calculate_circadian_metrics | `calculate_circadian_metrics(history; dt)` | Returns CircadianMetrics, cbtmin_times is Vector |
| circadian_metrics_from_events | From flat event list | Returns CircadianMetrics |
| sleep_metrics_from_events | From onset/offset events | Returns SleepMetrics |
| build_metrics_from_events | Batch metrics from events | Length 5 for circadian and sleep |
| baseline_cbtmin_from_events | `baseline_cbtmin_from_events(events, N, baseline_hrs)` | Returns (trajectories, counts), length 5 |
| events_to_trajectories | `events_to_trajectories(events, N)` | Length N |
| cbtmin_events_to_trajectory_list | `cbtmin_events_to_trajectory_list(events, N)` | Returns Vector |
| create_particle_metadata | `create_particle_metadata(id, history; dt)` | Returns ParticleMetadata, particle_id correct |
| create_particle_population | `create_particle_population(PH; dt)` | Returns ParticlePopulation, 5 particles |
| calculate_population_statistics | `calculate_population_statistics(particles)` | Returns Dict |
| get_top_particles | `get_top_particles(pop, n)` | Returns n particles |
| get_particles_by_criteria | `get_particles_by_criteria(pop; min_fitness, min_consistency)` | Returns Vector{ParticleMetadata} |

---

### 9. particle_summaries_tests.jl

Tests Von Mises fitting and legacy particle summaries.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| estimate_VonMeises | `estimate_VonMeises(phases)` | Returns (mu, Rbar, K); Rbar∈[0,1], K≥0 |
| particle_summaries | `particle_summaries(PH, t)` | Each summary has :CBTmin_mle when non-empty |

---

### 10. mle_summaries_tests.jl

Tests MLE-based population summaries, analysis, and I/O.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| compute_mle_population_summaries | With relaxed min_sleep_episodes | Each summary has tau, K, CBTmin_mle, sleep_onset_mle, sleep_duration_mle; tau∈(20,28) |
| analyze_mle_summaries | `analyze_mle_summaries(summaries)` | Returns Dict |
| save_mle_summaries | `save_mle_summaries(summaries, path)` | File created |
| load_mle_summaries | `load_mle_summaries(path)` | Loaded summaries match original length |

---

### 11. summary_histograms_tests.jl

Tests MLE summary histogram plotting.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| plot_summary_histograms | `plot_summary_histograms(summaries)` | Non-empty summaries produced, plot is non-nothing, saves PNG |

---

### 12. jetlag_integration_tests.jl

End-to-end jet lag pipeline tests (light diets, timezone shifts, steady-state detection).

---

### 13. steady_state_tests.jl

Tests steady-state detection and cyclic time utilities.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| unwrap24 | `unwrap24(times)` | Removes 24h wraparound jumps |
| wrap24 | `wrap24(val)` | Maps values into [0, 24) |
| is_steady_state_window | `is_steady_state_window(vals, threshold)` | Detects stability |
| find_baseline_steady_state_cbtmin | `find_baseline_steady_state_cbtmin(trajs)` | Extracts baseline CBTmin |
| find_first_steady_state_day | `find_first_steady_state_day(vals, ...)` | Detects first steady day |
| compute_adaptation_time | `compute_adaptation_time(vals, ...)` | Computes adaptation time |
| normalize_time! | `normalize_time!(times)` | Shifts ≥12h by −24 (mutating) |
| normalize_time | `normalize_time(times)` | Non-mutating version |

---

### 14. threading_tests.jl

Tests threading utilities.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| check_threading_status | `check_threading_status()` | No error |
| get_optimal_thread_count | `get_optimal_thread_count()` | Int ≥ 1 |
| threading_summary | `threading_summary()` | No error |

---

### 15. plot_tests.jl

Visual verification: generates plots and saves to `test/output/` for inspection.

| Test | Function / Concept | Assertions |
|------|-------------------|------------|
| plot_population_with_neuronal_dynamics | `plot_population_with_neuronal_dynamics(PH)` | Returns non-nothing, saves `population_neuronal_dynamics.png` |
| plot_summary_histograms | `plot_summary_histograms(summaries)` | Returns non-nothing when summaries non-empty, saves `summary_histograms.png` |

---

## Test Utilities

**`test_utils.jl`**

| Function | Purpose |
|----------|---------|
| `test_output_dir(parts...)` | Returns path to `test/output/` (or subpath), creates directory if needed |

---

## Output Files

| File | Source | Description |
|------|--------|-------------|
| `test/output/population_neuronal_dynamics.png` | plot_tests.jl | Neuronal dynamics plot for particle population |
| `test/output/summary_histograms.png` | plot_tests.jl | MLE summary histograms |
| `test/output/test_summaries.jld2` | mle_summaries_tests.jl | Saved/loaded MLE summaries |

---

## Coverage Summary

The test suite covers:

- **Model**: Parameters, space types, default states/actions/observations
- **Initialization**: Phased, random, custom; single location; edge cases (N=0, N=1, invalid mode)
- **Light diets**: Basic, shifted, period, intervention; Regular_V4_061hEarlier (on-the-fly); timezone helpers
- **Transitions**: `transition`, sleep-aware, ext, and circadian-only paths
- **Detection**: CBTmin (count validation), sleep periods (onset < offset, duration > 0)
- **Simulation**: Full population, events-only (memory-efficient)
- **Metadata**: Sleep/circadian metrics, event-based metrics, particle/population metadata
- **Summaries**: Von Mises, legacy summaries, MLE summaries with analysis, print, and I/O
- **Steady state**: unwrap24, wrap24, normalize_time, is_steady_state_window, adaptation time
- **Threading**: Status and optimal thread count (optional tests)
- **Plotting**: Population dynamics and summary histograms

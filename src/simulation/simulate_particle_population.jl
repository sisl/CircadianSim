"""
Simulate a population of particles through a sequence of actions.

# Arguments
- `N`: Number of particles
- `actions`: Sequence of `AbstractCircadianAction` (e.g. `CircadianAction` or `CircadianActionExt`)
- `params`: Model parameters from `model_parameters()`
- `init_fn`: (N) -> initial particles (default: sampled particles at t0)
- `transition_fn`: (state, action, params) -> next state (default: `transition`)

# Returns
- `particle_histories`: `Vector{Vector{CircadianState}}` — one trajectory per particle
"""
function simulate_particle_population(N::Int, actions, params; 
                                     init_fn=(N -> initialize_particles(N, DateTime(2023, 1, 1); state_params_mode=:sampled)), 
                                     transition_fn=transition,
                                     seed::Union{Int,Nothing}=nothing)
    initial_particles = init_fn(N)
    
    num_timesteps = length(actions) + 1
    particle_histories = [Vector{CircadianState}(undef, num_timesteps) for _ in 1:N]

    for i in 1:N
        particle_histories[i][1] = initial_particles[i]
    end

    rngs = seed !== nothing ? [StableRNG(seed + i) for i in 1:N] : nothing

    for (t, a) in enumerate(actions)
        Threads.@threads for i in 1:N
            rng_i = rngs !== nothing ? rngs[i] : Random.default_rng()
            s_new = transition_fn(particle_histories[i][t], a, params; rng=rng_i)
            particle_histories[i][t+1] = s_new
        end
    end

    return particle_histories
end

"""
    simulate_particle_population_explicit(N, actions, params, init_fn, transition_fn)

!!! warning "Deprecated"
    Use `simulate_particle_population(N, actions, params; init_fn, transition_fn)` instead.
"""
function simulate_particle_population_explicit(N::Int, actions, params, 
                                              init_fn, transition_fn;
                                              seed::Union{Int,Nothing}=nothing)
    initial_particles = init_fn(N)
    
    num_timesteps = length(actions) + 1
    particle_histories = [Vector{CircadianState}(undef, num_timesteps) for _ in 1:N]

    for i in 1:N
        particle_histories[i][1] = initial_particles[i]
    end

    rngs = seed !== nothing ? [StableRNG(seed + i) for i in 1:N] : nothing

    for (t, a) in enumerate(actions)
        Threads.@threads for i in 1:N
            rng_i = rngs !== nothing ? rngs[i] : Random.default_rng()
            s_new = transition_fn(particle_histories[i][t], a, params; rng=rng_i)
            particle_histories[i][t+1] = s_new
        end
    end

    return particle_histories
end

"""
    simulate_particle_from_initial_population(initial_particles, actions, params, transition_fn; seed=nothing, force_threading=false) -> Vector{Vector{CircadianState}}

Simulate from an existing particle ensemble through a sequence of actions; returns full state histories.

Use when you need per-step state trajectories (e.g. debugging or custom analysis). For JetLag adaptation-time
or MLE pipelines, prefer `simulate_particle_population_events_only` to save memory.

# Arguments
- `initial_particles`: `Vector{CircadianState}` from e.g. `initialize_particles(N, t0; state_params_mode=:sampled_normal)`.
- `actions`: `Vector{CircadianAction}` (precomputed) or on-the-fly action sequence for `CircadianActionExt`.
- `params`: from `model_parameters()`.
- `transition_fn`: `circadian_transition_sleep` for precomputed light; `circadian_transition_ext` for on-the-fly light.
- `seed`: optional `Int` for reproducible simulation. Each particle gets a deterministic `StableRNG(hash(seed, pid))`,
  making results identical regardless of thread count or scheduling.

# Returns
- `Vector{Vector{CircadianState}}`: one trajectory per particle (initial state + one state per action step).
"""
function simulate_particle_from_initial_population(initial_particles, actions, params, transition_fn; seed::Union{Int,Nothing}=nothing, force_threading::Bool=false)
    
    N = length(initial_particles)
    
    StateType = typeof(initial_particles[1])
    
    num_timesteps = length(actions) + 1
    particle_histories = [Vector{StateType}(undef, num_timesteps) for _ in 1:N]
    
    for i in 1:N
        particle_histories[i][1] = initial_particles[i]
    end
    
    use_threading = force_threading || (N >= 50 && Threads.nthreads() > 1)

    rngs = seed !== nothing ? [StableRNG(seed + i) for i in 1:N] : nothing

    for (t, a) in enumerate(actions)
        if use_threading
            Threads.@threads for i in 1:N
                rng_i = rngs !== nothing ? rngs[i] : Random.default_rng()
                s_new = transition_fn(particle_histories[i][t], a, params; rng=rng_i)
                particle_histories[i][t+1] = s_new
            end
        else
            for i in 1:N
                rng_i = rngs !== nothing ? rngs[i] : Random.default_rng()
                s_new = transition_fn(particle_histories[i][t], a, params; rng=rng_i)
                particle_histories[i][t+1] = s_new
            end
        end
    end
    
    return particle_histories
end

# ============================================================================
# ON-THE-FLY SIMULATION (events only, no full history storage)
# ============================================================================
# Simulates particles and captures CBTmin + sleep onset/offset events during
# the forward pass. Uses O(events) memory instead of O(N×T), particle-parallel
# threading for better cache locality, and linear interpolation for sub-step
# CBTmin timing precision.
#
# Use when you only need circadian/sleep metrics (e.g. CBTmin clock times,
# sleep onset/offset) and not full state trajectories.
# ============================================================================

"""
    simulate_particle_population_events_only(initial_particles, actions, params, transition_fn, start_date::DateTime; dt=1/60, verbose=true, seed=nothing)

Simulate particles and capture CBTmin + sleep onset/offset events on-the-fly.
Does not store full state histories—only event lists—so memory is O(events) instead of O(N×T).

Use for JetLag: pass `circadian_transition_sleep` when `actions` is a `Vector{CircadianAction}` (precomputed
light diet); use `circadian_transition_ext` for on-the-fly light with `CircadianActionExt`.

# Keywords
- `seed::Union{Int,Nothing}=nothing`: when provided, each particle gets a deterministic `StableRNG(hash(seed, pid))`,
  making results identical regardless of thread count or scheduling order.
- `return_final_states::Bool=false`: also return each particle's state after the last action (for continuing a
  simulation from where this one ended, e.g. branching scenarios from a shared baseline) without storing
  full state histories.

# Returns
- `(cbtmin_events, sleep_onset_events, sleep_offset_events)` where each is a vector of named tuples:
  `(particle_id, step_idx, clock_h, t_hours, datetime)`.
- With `return_final_states=true`: `(cbtmin_events, sleep_onset_events, sleep_offset_events, final_states)`,
  where `final_states[pid]` is particle `pid`'s state after the last action.
"""
function simulate_particle_population_events_only(
    initial_particles,
    actions,
    params,
    transition_fn,
    start_date::DateTime;
    dt=1/60,
    verbose=true,
    use_threading::Bool=true,
    force_threading::Bool=false,
    seed::Union{Int,Nothing}=nothing,
    return_final_states::Bool=false,
)
    N = length(initial_particles)
    final_states = return_final_states ? Vector{eltype(initial_particles)}(undef, N) : nothing

    # CBTmin detection — calibrated constants defined in sleep_CBTmin_detection.jl
    p_ref  = CBTMIN_TEMPORAL_OFFSET_H

    cbtmin_events = Dict{Int, Vector{Any}}()
    cbtmin_events_lock = ReentrantLock()
    sleep_onset_events = Dict{Int, Vector{Any}}()
    sleep_offset_events = Dict{Int, Vector{Any}}()
    sleep_events_lock = ReentrantLock()

    # Use threading when: (force_threading and nthreads>1) or (N>=50 and nthreads>1)
    use_threading = use_threading && (Threads.nthreads() > 1) && (force_threading || N >= 50)

    if verbose
        println("  Processing $(N) particles with on-the-fly event capture...")
        if use_threading
            println("  Using $(Threads.nthreads()) threads (particle-parallel)")
        end
        println("  Sub-step CBTmin precision via linear interpolation")
    end

    total_crossings = Threads.Atomic{Int}(0)
    total_sleep_onsets = Threads.Atomic{Int}(0)
    total_sleep_offsets = Threads.Atomic{Int}(0)

    _simulate_events_loop!(
        initial_particles,
        actions,
        params,
        transition_fn,
        start_date,
        dt,
        p_ref,
        cbtmin_events,
        cbtmin_events_lock,
        sleep_onset_events,
        sleep_offset_events,
        sleep_events_lock,
        total_crossings,
        total_sleep_onsets,
        total_sleep_offsets,
        use_threading,
        verbose,
        seed;
        final_states=final_states,
    )

    all_cbtmin = vcat(values(cbtmin_events)...)
    all_sleep_onset = vcat(values(sleep_onset_events)...)
    all_sleep_offset = vcat(values(sleep_offset_events)...)

    sort!(all_cbtmin, by=e -> (e.particle_id, e.t_hours))
    sort!(all_sleep_onset, by=e -> (e.particle_id, e.t_hours))
    sort!(all_sleep_offset, by=e -> (e.particle_id, e.t_hours))

    if verbose
        println("  CBTmin events: $(length(all_cbtmin)), sleep onset: $(length(all_sleep_onset)), sleep offset: $(length(all_sleep_offset))")
    end

    return_final_states && return all_cbtmin, all_sleep_onset, all_sleep_offset, final_states
    return all_cbtmin, all_sleep_onset, all_sleep_offset
end

function _simulate_events_loop!(
    initial_particles,
    actions,
    params,
    transition_fn,
    start_date,
    dt,
    p_ref,
    cbtmin_events,
    cbtmin_events_lock,
    sleep_onset_events,
    sleep_offset_events,
    sleep_events_lock,
    total_crossings,
    total_sleep_onsets,
    total_sleep_offsets,
    use_threading,
    verbose,
    seed::Union{Int,Nothing}=nothing;
    final_states::Union{Nothing,AbstractVector}=nothing,
)
    N = length(initial_particles)
    total_t_hours = length(actions) * dt

    _run_particle = function(pid)
        local_cbtmin = Any[]
        local_sleep_onset = Any[]
        local_sleep_offset = Any[]

        local_rng = seed !== nothing ? StableRNG(seed + pid) : Random.default_rng()

        state = initial_particles[pid]
        prev_phase = atan(state.xc, state.x)
        prev_sleep = state.sleep
        pending_cbtmin = Float64[]

        for (t, action) in enumerate(actions)
            next_state = transition_fn(state, action, params; rng=local_rng)
            current_t_hours = t * dt

            # Sleep transitions
            if !prev_sleep && next_state.sleep
                Threads.atomic_add!(total_sleep_onsets, 1)
                clock_h = mod(current_t_hours, 24.0)
                datetime_val = start_date + Dates.Millisecond(round(Int, current_t_hours * 3600 * 1000))
                push!(local_sleep_onset, (
                    particle_id=pid,
                    step_idx=t + 1,
                    clock_h=clock_h,
                    t_hours=current_t_hours,
                    datetime=datetime_val
                ))
            elseif prev_sleep && !next_state.sleep
                Threads.atomic_add!(total_sleep_offsets, 1)
                clock_h = mod(current_t_hours, 24.0)
                datetime_val = start_date + Dates.Millisecond(round(Int, current_t_hours * 3600 * 1000))
                push!(local_sleep_offset, (
                    particle_id=pid,
                    step_idx=t + 1,
                    clock_h=clock_h,
                    t_hours=current_t_hours,
                    datetime=datetime_val
                ))
            end

            # Flush pending CBTmin events
            while !isempty(pending_cbtmin) && pending_cbtmin[1] <= current_t_hours
                cbtmin_t_hours = popfirst!(pending_cbtmin)
                if cbtmin_t_hours <= total_t_hours
                    clock_h = mod(cbtmin_t_hours, 24.0)
                    datetime_val = start_date + Dates.Millisecond(round(Int, cbtmin_t_hours * 3600 * 1000))
                    push!(local_cbtmin, (
                        particle_id=pid,
                        step_idx=round(Int, cbtmin_t_hours / dt) + 1,
                        clock_h=clock_h,
                        t_hours=cbtmin_t_hours,
                        datetime=datetime_val
                    ))
                end
            end

            # Phase crossing -> CBTmin
            current_phase = atan(next_state.xc, next_state.x)
            crossing_fraction = cbtmin_phase_crossing(prev_phase, current_phase)
            if crossing_fraction !== nothing
                Threads.atomic_add!(total_crossings, 1)
                crossing_t_hours = (t - 1) * dt + crossing_fraction * dt
                cbtmin_t_hours = crossing_t_hours + p_ref
                if cbtmin_t_hours <= total_t_hours
                    insert_idx = searchsortedfirst(pending_cbtmin, cbtmin_t_hours)
                    insert!(pending_cbtmin, insert_idx, cbtmin_t_hours)
                end
            end

            state = next_state
            prev_phase = current_phase
            prev_sleep = next_state.sleep
        end

        # Remaining pending CBTmin
        while !isempty(pending_cbtmin)
            cbtmin_t_hours = popfirst!(pending_cbtmin)
            if cbtmin_t_hours <= total_t_hours
                clock_h = mod(cbtmin_t_hours, 24.0)
                datetime_val = start_date + Dates.Millisecond(round(Int, cbtmin_t_hours * 3600 * 1000))
                push!(local_cbtmin, (
                    particle_id=pid,
                    step_idx=round(Int, cbtmin_t_hours / dt) + 1,
                    clock_h=clock_h,
                    t_hours=cbtmin_t_hours,
                    datetime=datetime_val
                ))
            end
        end

        # Merge into shared dict under lock (once per particle)
        if !isempty(local_cbtmin)
            lock(cbtmin_events_lock) do
                thread_id = Threads.threadid()
                vec = get!(cbtmin_events, thread_id, Vector{Any}())
                append!(vec, local_cbtmin)
            end
        end
        if !isempty(local_sleep_onset)
            lock(sleep_events_lock) do
                thread_id = Threads.threadid()
                vec = get!(sleep_onset_events, thread_id, Vector{Any}())
                append!(vec, local_sleep_onset)
            end
        end
        if !isempty(local_sleep_offset)
            lock(sleep_events_lock) do
                thread_id = Threads.threadid()
                vec = get!(sleep_offset_events, thread_id, Vector{Any}())
                append!(vec, local_sleep_offset)
            end
        end

        final_states === nothing || (final_states[pid] = state)   # each pid written by one task only

        if verbose && pid % 500 == 0
            println("    Thread $(Threads.threadid()): processed $pid / $N particles")
        end
    end

    if use_threading
        Threads.@threads for pid in 1:N
            _run_particle(pid)
        end
    else
        for pid in 1:N
            _run_particle(pid)
        end
    end
end
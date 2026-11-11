"""
    generate_light_diet_basic(; days, light_on, light_off, lux_on, lux_off, sleep_light, sleep_dark) -> Vector{CircadianAction}

Build a precomputed light diet: one `CircadianAction` per minute for `days` days.

Light is on from `light_on` to `light_off` (clock hours, 0–24) with `lux_on`; otherwise `lux_off`.
`sleep_light` / `sleep_dark` set `CircadianAction.sleep_forced` during those minutes:
`true` = forced sleep, `false` = forced wake, `missing` = homeostat (open).

Use with `circadian_transition_sleep` and `simulate_particle_population_events_only` for JetLag baseline
(e.g. `light_on=REGULAR_SCHEDULE_LIGHT_ON_H`, `light_off=REGULAR_SCHEDULE_LIGHT_OFF_H`, `lux_on=150` or `10_000`, `lux_off=0`).
"""
function generate_light_diet_basic(; days::Int, light_on::Float64, light_off::Float64,lux_on::Float64, lux_off::Float64, sleep_light::Union{Bool, Missing}, sleep_dark::Union{Bool, Missing})
    actions = CircadianAction[]
    dt = 1 / 60  # 1 minute in hours
    minutes_per_day = 24 * 60
    total_minutes = days * minutes_per_day

    for i in 1:total_minutes
        minute_of_day = mod(i - 1, minutes_per_day)
        is_light = minute_of_day >= light_on * 60 && minute_of_day < light_off * 60  #6 am start

        I_act = is_light ? lux_on : lux_off
        sleep_forced = is_light ? sleep_light : sleep_dark
        push!(actions, CircadianAction(I_act, sleep_forced, dt))
    end

    return actions
end

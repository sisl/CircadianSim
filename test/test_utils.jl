# Test utilities: output directory for visual verification plots

"""
Return path to test output directory (created if needed).
Plots and other visual verification outputs go here.
"""
function test_output_dir(parts::String...)
    d = joinpath(@__DIR__, "output", parts...)
    mkpath(d)
    return d
end
